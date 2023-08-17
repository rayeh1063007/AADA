import math

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from kornia.augmentation.base import AugmentationBase2D
from mmseg.models.utils.dacs_transforms import denorm,renorm
class RandomGaussianNoise(AugmentationBase2D):
    r"""Add gaussian noise to a batch of multi-dimensional images.

    Args:
        mean (float): The mean of the gaussian distribution. Default: 0.
        std (float): The standard deviation of the gaussian distribution. Default: 1.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        p (float): probability of applying the transformation. Default value is 0.5.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 2, 2)
        >>> RandomGaussianNoise(mean=0., std=1., p=1.)(img)
        tensor([[[[ 2.5410,  0.7066],
                  [-1.1788,  1.5684]]]])
    """

    def __init__(self,
                 mean: float = 0.,
                 std: float = 1.,
                 return_transform: bool = False,
                 same_on_batch: bool = False,
                 p: float = 0.5) -> None:
        super(RandomGaussianNoise, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, shape: torch.Size) -> Dict[str, torch.Tensor]:
        noise = torch.randn(shape)
        return dict(noise=noise)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return input + params['noise'].to(input.device) * self.std + self.mean


class TraditionAugmentation(nn.Module):
    def __init__(self, f_dim=512, n_dim=128, dropout_ratio=0.8, with_condition=True, init='random'):
        super().__init__()

        # affine transforms (mid, range)
        self.angle = [0.0, 30.0] # [-30.0:30.0] rotation angle 0
        self.trans = [0.0, 0.45] # [-0.45:0.45] X/Y translate 1, 2
        self.shear = [0.0, 0.30] # [-0.30:0.30] X/Y shear 3, 4
        self.sharpen = [1.0, 2.5, 2.5]
        # gaussian
        self.blur_kernel = [1, 2]
        self.blur_sigma = [2.6,2.5] # [0.1:5.1]
        self.noise = [0.25,0.25] # [0:10]
        # elastic_transform
        self.elt = [0.0, 5.0]
        # color transforms (mid, range)
        self.bri = [0.0, 1.0] # [-0.9:0.9] brightness
        self.con = [1.0, 2.5, 2.5] # [0.1:1.9] contrast
        self.sat = [1.0, 2.5, 2.5] # [-0.30:0.30] saturation
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                                        [0.07, 0.99, 0.11],
                                        [0.27, 0.57, 0.78]], dtype=torch.float32)
        self.hed_from_rgb = torch.inverse(self.rgb_from_hed)

        hidden = 512
        linear = lambda ic, io : nn.Linear(ic, io, False)
        conv = lambda ic, io, k : nn.Conv2d(ic, io, k, padding=k//2, bias=False)
        bn1d = lambda c : nn.BatchNorm1d(c, track_running_stats=False)
        bn2d = lambda c : nn.BatchNorm2d(c, track_running_stats=False)
        pool = lambda k, s : nn.MaxPool2d(kernel_size=k, stride=s)

        if with_condition:
            self.context_enc_body = nn.Sequential(
                conv(f_dim,256,1),
                bn2d(256),
                nn.LeakyReLU(0.2, True),
                pool(4, 2),
                conv(256,64,1),
                bn2d(64),
                nn.LeakyReLU(0.2, True),
                pool(4, 3),
            )

        self.body = nn.Sequential(
            linear(n_dim + 256 if with_condition else n_dim, hidden),
            bn1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            linear(hidden, hidden),
            bn1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
        )

        self.p_regressor = linear(hidden, 15)
        self.m_regressor = linear(hidden, 18)

        self.actP = nn.Sigmoid()
        self.actM = nn.Sigmoid()

        # initialize parameters
        self.reset(init)

        self.with_condition = with_condition

        self.relax = True
        self.stochastic = True

    def rgb_to_hed(self, rgb):
        r: torch.Tensor = rgb[..., 0, :, :]
        g: torch.Tensor = rgb[..., 1, :, :]
        b: torch.Tensor = rgb[..., 2, :, :]
        m = self.hed_from_rgb
        h: torch.Tensor = m[0,0] * r + m[0,1] * g + m[0,2] * b
        e: torch.Tensor = m[1,0] * r + m[1,1] * g + m[1,2] * b
        d: torch.Tensor = m[2,0] * r + m[2,1] * g + m[2,2] * b

        out: torch.Tensor = torch.stack([h, e, d], -3)

        return out

    def hed_to_rgb(self, stains):
        h: torch.Tensor = stains[..., 0, :, :]
        e: torch.Tensor = stains[..., 1, :, :]
        d: torch.Tensor = stains[..., 2, :, :]
        m = self.rgb_from_hed
        r: torch.Tensor = m[0,0] * h + m[0,1] * e + m[0,2] * d
        g: torch.Tensor = m[1,0] * h + m[1,1] * e + m[1,2] * d
        b: torch.Tensor = m[2,0] * h + m[2,1] * e + m[2,2] * d
        
        out: torch.Tensor = torch.stack([r, g, b], -3)

        return out

    def forward(self, noise, c=None):
        if self.with_condition:
            c = self.context_enc_body(c)
            c = torch.flatten(c, 1)
            features = torch.cat((c, noise), dim=1)
        else:
            features = noise
        features = self.body(features)
        p_stratgy = self.actP(self.p_regressor(features))
        m_stratgy = self.actM(self.m_regressor(features))
        return p_stratgy, m_stratgy

    def reset(self, init):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if init=='random':
            nn.init.normal_(self.p_regressor.weight)
            nn.init.normal_(self.m_regressor.weight)
        elif init=='constant':
            nn.init.constant_(self.p_regressor.weight, 0.)
            nn.init.constant_(self.m_regressor.weight, 0.)

    def transform(self, x, x_t, means, stds, p_stratgy, m_stratgy, x_t_q=None):
        device = x.device
        B,C,H,W = x.shape
        # reparametrize probabilities and magnitudes
        paramP = p_stratgy.unsqueeze(1).permute((0,2,1)).repeat(1,1,2)
        paramP[:,:,1] = 1.0-paramP[:,:,1]
        sampleP = F.gumbel_softmax(paramP, tau=1.0, hard=True)
        sampleP = sampleP[:,:,0].permute(1,0)
        sampleM = torch.tanh(m_stratgy).permute(1,0)
        ############## color ###################
        x = denorm(x, means, stds)
        ##### equalize[[0],[]]
        x_equ = kornia.enhance.equalize(x)
        EQU = x - x_equ
        for i in range(B):
            EQU[i] = EQU[i].clone()* sampleP[0,i]
        x = x - EQU
        x = torch.clamp(x,min=0.0,max=1.0)
        ##### brightness[[1],[0]], contrast[[2],[1]], saturation[[3],[2]]
        BRI: torch.tensor = sampleP[1] * sampleM[0] 
        CON: torch.tensor = self.con[0] + sampleP[2] * (self.con[1] + sampleM[1] * self.con[2]) # mid + B(0/1)*U[0,1]*M/10
        SAT: torch.tensor = self.sat[0] + sampleP[3] * (self.sat[1] + sampleM[2] * self.sat[2])
        for i in range(B):
            x[i] = kornia.enhance.adjust_saturation(x[i].clone(), SAT[i])
            x[i] = torch.clamp(x[i].clone(),min=0.0,max=1.0)
            x[i] = kornia.enhance.adjust_brightness(x[i].clone(), BRI[i])
            x[i] = torch.clamp(x[i].clone(),min=0.0,max=1.0)
            x[i] = kornia.enhance.adjust_contrast(x[i].clone(), CON[i])
            x[i] = torch.clamp(x[i].clone(),min=0.0,max=1.0)
        ##### HSV[[4],[3,4,5]]
        x = kornia.color.rgb_to_hsv(x)
        for i in range(B):
            # Augment the hue channel.
            x[i, 0, :, :] = x[i, 0, :, :].clone() / (2.*math.pi)
            x[i, 0, :, :] = x[i, 0, :, :].clone() + (sampleM[3,i] % 1.0) * sampleP[4,i]
            x[i, 0, :, :] = x[i, 0, :, :].clone() % 1.0
            x[i, 0, :, :] = x[i, 0, :, :].clone() * (2.*math.pi)
            # Augment the Saturation channel.
            if sampleM[4,i] < 0.0:
                x[i, 1, :, :] = x[i, 1, :, :].clone() * (1.0 + sampleM[4,i] * sampleP[4,i])
            else:
                x[i, 1, :, :] = x[i, 1, :, :].clone() * (1.0 + (1.0 - x[i, 1, :, :]) * sampleM[4,i] * sampleP[4,i])
            # Augment the Brightness channel.
            if sampleM[5,i] < 0.0:
                x[i, 2, :, :] = x[i, 2, :, :].clone() * (1.0 + sampleM[5,i] * sampleP[4,i])
            else:
                x[i, 2, :, :] = x[i, 2, :, :].clone() + (1.0 - x[i, 2, :, :]) * sampleM[5,i] * sampleP[4,i]
        x = kornia.color.hsv_to_rgb(x)
        x = torch.clamp(x,min=0.0,max=1.0)
        ##### HED[[5],[6,7,8]]
        x = self.rgb_to_hed(x,)
        for i in range(B):
            x[i, 0, :, :] = x[i, 0, :, :].clone() * (1.0 + sampleM[6,i] * sampleP[5,i])
            x[i, 0, :, :] = x[i, 0, :, :].clone() + (torch.rand(1).to(device) * sampleM[6,i] * sampleP[5,i])
            x[i, 1, :, :] = x[i, 1, :, :].clone() * (1.0 + sampleM[7,i] * sampleP[5,i])
            x[i, 1, :, :] = x[i, 1, :, :].clone() + (torch.rand(1).to(device) * sampleM[7,i] * sampleP[5,i])
            x[i, 2, :, :] = x[i, 2, :, :].clone() * (1.0 + sampleM[8,i] * sampleP[5,i])
            x[i, 2, :, :] = x[i, 2, :, :].clone() + (torch.rand(1).to(device) * sampleM[8,i] * sampleP[5,i])
        x = self.hed_to_rgb(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        ##### gaussian blur[[6],[9]]
        BLUR_K: torch.tensor = self.blur_kernel[0] + sampleP[6] * self.blur_kernel[1]
        BLUR_S: torch.tensor = self.blur_sigma[0] + sampleM[9] * self.blur_sigma[1]
        for i in range(B):
            x[i] = kornia.filters.gaussian_blur2d(x[i].clone().unsqueeze(0), (int(BLUR_K[i]), int(BLUR_K[i])), (BLUR_S[i], BLUR_S[i]))
        x = torch.clamp(x,min=0.0,max=1.0)
        ##### sharpen[[7],[10]]
        SHARP: torch.tensor = self.sharpen[0] + sampleP[7] * (self.sharpen[1] + self.sharpen[2] * sampleM[10])
        x = kornia.enhance.sharpness(x, SHARP)
        x = torch.clamp(x,min=0.0,max=1.0)
        ##### gaussian noise[[8],[11]]
        NOISE: torch.tensor = sampleP[8] * (self.noise[0] + sampleM[11] * self.noise[1])
        for i in range(B):
            GN = RandomGaussianNoise(std=NOISE[i], p=1.)
            x[i] = torch.clamp(GN(x[i].clone().unsqueeze(0)), min=0.0, max=1.0)
        x = renorm(x, means, stds)
        ################# affine augmentations #################
        ##### elastic_transform[[9],[12]]
        ELT: torch.tensor = self.elt[0] + sampleP[9] * sampleM[12] * self.elt[1]
        if x_t_q is not None:
            for i in range(B):
                elt_ops = kornia.augmentation.AugmentationSequential(
                    kornia.augmentation.RandomElasticTransform(alpha=(ELT[i],ELT[i]),mode='nearest',p=1.),
                    data_keys=['input', 'mask', 'mask'],
                )
                x[i], x_t[i], x_t_q[i] = elt_ops(x[i].clone(),x_t[i].clone().to(torch.float),x_t_q[i].clone().to(torch.float))
        else:
            for i in range(B):
                elt_ops = kornia.augmentation.AugmentationSequential(
                    kornia.augmentation.RandomElasticTransform(alpha=(ELT[i],ELT[i]),mode='nearest',p=1.),
                    data_keys=['input', 'mask'],
                )
                x[i], x_t[i] = elt_ops(x[i].clone(),x_t[i].clone().to(torch.float))
        x = torch.clamp(x,min=0.0,max=1.0)
        ##### affine augmentations
        R: torch.tensor = torch.zeros(B,3,3).to(device) + torch.eye(3).to(device) #3*3對角線為1
        # define the rotation angle[[10],[13]]
        ANG: torch.tensor = sampleP[10] * sampleM[13] * self.angle[1] # B(0/1)*U[0,1]*M/10
        # define the rotation center
        CTR: torch.tensor = torch.cat((W*torch.ones(B).to(device)//2, H*torch.ones(B).to(device)//2)).view(-1,2)
        # define the scale factor
        SCL: torch.tensor = torch.ones((B, 2)).to(device)
        R[:,0:2] = kornia.geometry.transform.get_rotation_matrix2d(CTR, ANG, SCL)
        # translation: border not defined yet [[11,12],[14,15]]
        T: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(device)
        T[:,0,2] = W * sampleP[11] * sampleM[14] * self.trans[1]
        T[:,1,2] = H * sampleP[12] * sampleM[15] * self.trans[1]
        # shear: check this [[13,14],[16,17]]
        S: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(device)
        S[:,0,1] = sampleP[13] * sampleM[16] * self.shear[1]
        S[:,1,0] = sampleP[14] * sampleM[17] * self.shear[1]
        ##### apply the transformation to original image
        M: torch.tensor = torch.bmm(torch.bmm(S,T),R)
        x = kornia.geometry.transform.warp_perspective(x, M, dsize=(H,W), mode='bilinear', padding_mode ='zeros')
        x = torch.clamp(x,min=0.0,max=1.0)
        ones_mask = torch.ones_like(x_t)
        inv_ones_mask = kornia.geometry.transform.warp_perspective(ones_mask.to(torch.float), M, dsize=(H,W), mode='nearest', padding_mode="zeros") - 1
        inv_color_mask = inv_ones_mask * (-255)
        x_t = kornia.geometry.transform.warp_perspective(x_t.to(torch.float), M, dsize=(H,W), mode='nearest', padding_mode ='zeros') + inv_color_mask
        x_t = x_t.squeeze(1)
        if x_t_q is not None:
            x_t_q = kornia.geometry.transform.warp_perspective(x_t_q, M, dsize=(H,W), mode='nearest', padding_mode="zeros")
        x_t = x_t.to(torch.long).unsqueeze(1)
        return x, x_t.to(torch.long), x_t_q
