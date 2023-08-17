from . import augmentation_container
from . import cutout
from . import imagenet_augmentation
from . import nn_aug
from . import nn_aug_gt
from . import nn_aug_autodo
# from . import replay_buffer

def build_augmentation(f_dim, g_use, c_use, t_use, g_scale, c_scale, c_reg_coef=0, with_condition=True, init='random'):
    c_aug, g_aug, t_aug = None, None, None
    if t_use:
        t_aug = nn_aug_autodo.TraditionAugmentation(f_dim, with_condition=with_condition, init=init)
    else:
        if g_use: 
            if f_dim==512:
                g_aug = nn_aug.GeometricAugmentation(f_dim, g_scale, with_condition=with_condition, init=init)
            elif f_dim==19:
                g_aug = nn_aug_gt.GeometricAugmentation(f_dim, g_scale, with_condition=with_condition, init=init)
        if c_use: 
            if f_dim==512:
                c_aug = nn_aug.ColorAugmentation(f_dim, c_scale, with_condition=with_condition, init=init)
            elif f_dim==19:
                c_aug = nn_aug_gt.ColorAugmentation(f_dim, c_scale, with_condition=with_condition, init=init)
    # 將自動擴增模型(g/c/t)包裝成方便更新的container
    augmentation = augmentation_container.AugmentationContainer(c_aug, g_aug, t_aug, c_reg_coef, f_dim)
    return augmentation
