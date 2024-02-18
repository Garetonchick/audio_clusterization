import torch.nn as nn
from torchvision.transforms import v2
from byol_a import augmentations

class BatchAugmenter(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self.aug = aug
    
    def forward(self, x):
        B, C, H, W = x.shape
        return self.aug(x.view(B * C, H, W)).view(B, C, H, W)

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
    
    return BatchAugmenter(v2.Compose(augs))