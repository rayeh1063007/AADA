import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def relaxed_bernoulli(logits, temp=0.0001, device='cpu'):
    u = torch.rand_like(logits, device=device)
    l = torch.log(u) - torch.log(1 - u)
    return ((l + logits)/temp).sigmoid().to(int)

class TriangleWave(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        o = torch.acos(torch.cos(x * math.pi)) / math.pi
        self.save_for_backward(x)
        return o

    @staticmethod
    def backward(self, grad):
        o = self.saved_tensors[0]
        # avoid nan gradient at the peak by replacing it with the right derivative
        o = torch.floor(o) % 2
        grad[o == 1] *= -1 
        return grad


class ColorAugmentation(nn.Module):
    def __init__(self, f_dim=19, scale=1, hidden=128, n_dim=128, dropout_ratio=0.8, with_condition=True, init='random'):
        super().__init__()
        
        n_hidden = 4 * n_dim
        conv = lambda ic, io, k : nn.Conv2d(ic, io, k, padding=k//2, bias=False)
        linear = lambda ic, io : nn.Linear(ic, io, False)
        bn2d = lambda c : nn.BatchNorm2d(c, track_running_stats=False)
        bn1d = lambda c : nn.BatchNorm1d(c, track_running_stats=False)

        # embedding layer for context vector
        if with_condition:
            self.color_enc1 = conv(3+f_dim, hidden, 1)
        else:
            self.color_enc1 = conv(3, hidden, 1)
        # embedding layer for RGB
        # body for RGB
        self.color_enc_body = nn.Sequential(
            bn2d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            conv(hidden, hidden, 1),
            bn2d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Sequential()
        )
        # output layer for RGB
        self.c_regress = conv(hidden, 6, 1)
        # body for noise vector
        self.noise_enc = nn.Sequential(
            linear(n_dim, n_hidden),
            bn1d(n_hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            linear(n_hidden, n_hidden),
            bn1d(n_hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
        )
        # output layer for noise vector
        self.n_regress = linear(n_hidden, 2)

        self.register_parameter('logits', nn.Parameter(torch.zeros(1)))
        # initialize parameters
        self.reset(init)

        self.with_condition = with_condition
        self.f_dim = f_dim
        self.scale = scale
        self.relax = True
        self.stochastic = True

    def sampling(self, scale, shift, temp=0.0001):
        prob = torch.ones([scale.shape[0]])
        if self.stochastic: # random apply
            logits = self.logits.repeat(scale.shape[0]).reshape(-1, 1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=scale.device)
            if not self.relax: # hard sampling
                prob = (prob > 0.5).float()
            scale = 1 - prob + prob * scale
            shift = prob * shift # omit "+ (1 - prob) * 0"
        return scale, shift, prob.squeeze().to(int)

    def forward(self, x, noise, c=None, update=False):
        B,C,H,W = x.shape
        # per-pixel scale and shift "with" context information
        if self.with_condition:
            targets = c.clone()
            n_classes = self.f_dim
            targets[torch.where(targets==255)] = n_classes
            onehot_targets = F.one_hot(targets, n_classes+1).float()
            if onehot_targets.ndim==5:
                onehot_targets = onehot_targets[:,:,:,:,:-1]
                onehot_targets = onehot_targets.squeeze(1).permute(0,3,1,2)
            elif onehot_targets.ndim==4:
                onehot_targets = onehot_targets[:,:,:,:-1]
                onehot_targets = onehot_targets.permute(0,3,1,2)
            xc = torch.cat([x,onehot_targets],1)
            feature = self.color_enc1(xc)
        else: # per-pixel scale and shift "without" context information
            feature = self.color_enc1(x)
        feature = self.color_enc_body(feature)
        factor = self.c_regress(feature)
        gfactor = self.noise_enc(noise)
        gfactor = self.n_regress(gfactor).reshape(-1, 2, 1, 1)
        # add up parameters
        scale, shift = factor.chunk(2, dim=1)
        g_scale, g_shift = gfactor.chunk(2, dim=1)
        scale = (g_scale + scale).sigmoid()
        shift = (g_shift + shift).sigmoid()
        # scaling
        scale = self.scale * (scale - 0.5) + 1
        shift = shift - 0.5
        # random apply
        if update:
            prob = torch.ones([scale.shape[0]])
        else:
            scale, shift, prob = self.sampling(scale, shift)

        return scale, shift, prob

    def reset(self,init):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if init=='random':
            nn.init.normal_(self.c_regress.weight, 0)
            nn.init.normal_(self.n_regress.weight, 0)
        elif init=='constant':
            nn.init.constant_(self.c_regress.weight, 0)
            nn.init.constant_(self.n_regress.weight, 0)
        nn.init.constant_(self.logits, 0)

    def transform(self, x, scale, shift):
        # ignore zero padding region
        with torch.no_grad():
            h, w = x.shape[-2:]
            mask = (x.sum(1, keepdim=True) == 0).float() # mask pixels having (0, 0, 0) color
            mask = torch.logical_and(mask.sum(-1, keepdim=True) < w,
                                     mask.sum(-2, keepdim=True) < h) # mask zero padding region

        x = (scale * x + shift) * mask
        # return TriangleWave.apply(x)
        return x
        

class GeometricAugmentation(nn.Module):
    def __init__(self, f_dim=19, scale=0.5, n_dim=128, dropout_ratio=0.8, with_condition=True, init='random'):
        super().__init__()

        hidden = 512
        linear = lambda ic, io : nn.Linear(ic, io, False)
        conv = lambda ic, io, k : nn.Conv2d(ic, io, k, padding=k//2, bias=False)
        bn1d = lambda c : nn.BatchNorm1d(c, track_running_stats=False)
        bn2d = lambda c : nn.BatchNorm2d(c, track_running_stats=False)
        mpool = lambda k, s : nn.MaxPool2d(kernel_size=k, stride=s)
        apool = lambda k, s : nn.AvgPool2d(kernel_size=k, stride=s)

        if with_condition:
            self.context_enc_body = nn.Sequential(
                conv(f_dim,128,1),
                bn2d(128),
                nn.LeakyReLU(0.2, True),
                mpool(4, 4),
                conv(128,64,1),
                bn2d(64),
                nn.LeakyReLU(0.2, True),
                mpool(4, 4),
                conv(64,1,1),
                bn2d(1),
                nn.LeakyReLU(0.2, True),
                apool(4, 4),
            )

        self.body = nn.Sequential(
            linear(n_dim + 256 if with_condition else n_dim, hidden),
            bn1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            linear(hidden, 128),
            bn1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
        )

        self.regressor = linear(128, 6)
        # identity matrix
        self.register_buffer('i_matrix', torch.Tensor([[1, 0, 0], [0, 1, 0]]).reshape(1, 2, 3))

        self.register_parameter('logits', nn.Parameter(torch.zeros(1)))
        # initialize parameters
        self.reset(init)

        self.with_condition = with_condition
        self.scale = scale
        self.f_dim = f_dim

        self.relax = True
        self.stochastic = True

    def sampling(self, A, temp=0.0001):
        if self.stochastic: # random apply
            logits = self.logits.repeat(A.shape[0]).reshape(-1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=logits.device)
            if not self.relax: # hard sampling
                prob = (prob > 0.5).float()
            return ((1 - prob) * self.i_matrix + prob * A), prob.squeeze().to(int)
        else:
            return A

    def forward(self, x, noise, c=None, update=False):
        if self.with_condition:
            targets = c.clone()
            n_classes = self.f_dim
            targets[torch.where(targets==255)] = n_classes
            onehot_targets = F.one_hot(targets, n_classes+1).float()
            onehot_targets = onehot_targets[:,:,:,:,:-1]
            onehot_targets = onehot_targets.squeeze(1).permute(0,3,1,2)
            features = self.context_enc_body(onehot_targets)
            features = torch.flatten(features, 1)
            features = torch.cat((noise,features), dim=1)
        else:
            features = noise
        features = self.body(features)
        A = self.regressor(features).reshape(-1, 2, 3)
        # scaling
        A = self.scale * (A.sigmoid() - 0.5) + self.i_matrix
        # random apply
        if update:
            prob = torch.ones([A.shape[0]])
        else:
            A, prob = self.sampling(A)
        # matrix to grid representation
        grid = nn.functional.affine_grid(A, x.shape)
        return grid, prob

    def reset(self, init='random'):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # zero initialization
        if init=='random':
            nn.init.normal_(self.regressor.weight, 0)
        elif init=='constant':
            nn.init.constant_(self.logits, 0)

    def transform(self, x, x_t, grid, pw=None):
        x = F.grid_sample(x, grid, mode='bilinear')
        with torch.no_grad():
            ones_mask = torch.ones_like(x_t)
            inv_ones_mask = F.grid_sample(ones_mask.to(torch.float), grid, mode='nearest', padding_mode="zeros") - 1
            inv_color_mask = inv_ones_mask * (-255)
            x_t = F.grid_sample(x_t.to(torch.float), grid, mode='nearest', padding_mode="zeros") + inv_color_mask
            if pw is not None:
                pw = F.grid_sample(pw.to(torch.float), grid, mode='nearest', padding_mode="zeros")
        return x, x_t.to(torch.long), pw
