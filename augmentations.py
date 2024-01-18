import torch.nn as nn
from torchvision.transforms import v2
from byol_a import augmentations

def get_augmentation(aug_cfg):
    aug_classes = [
        augmentations.RandomResizeCrop, 
        augmentations.RandomLinearFader, 
        augmentations.MixupBYOLA,
        nn.Identity
    ]
    aug_names = [aug.__name__ for aug in aug_classes]
    augs = []
    for aug_name in aug_cfg:
        if aug_name not in aug_names:
            raise ValueError(f'Unknown augmentation: "{aug_name}"')
        augs.append(aug_classes[aug_names.index(aug_name)]())
    
    return v2.Compose(augs)