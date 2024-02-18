import os
import sys

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(repo_dir, 'byol-a', 'v2'))

from byol_a2.common import load_yaml_config
from byol_a2.models import AudioNTT2022, load_pretrained_weights
from byol_a2.augmentations import PrecomputedNorm
from byol_a2 import augmentations
from copy import deepcopy

byola_cfg = load_yaml_config(os.path.join(repo_dir, 'byol-a', 'v2', 'config_v2.yaml'))

def get_byola_cfg():
    return deepcopy(byola_cfg)

def get_normalizer(dataset_stats):
    return PrecomputedNorm(dataset_stats)

def get_frozen_pretrained_byola(dataset_stats, device, only_highlevel_features=False):
    pretrained_byola = AudioNTT2022(d=byola_cfg.feature_d, n_mels=byola_cfg.n_mels)
    load_pretrained_weights(
        pretrained_byola, 
        os.path.join(repo_dir, 'byol-a', 'v2', 'AudioNTT2022-BYOLA-64x96d2048.pth')
    )
    pretrained_byola.eval()

    for param in pretrained_byola.parameters():
        param.requires_grad = False

    normalizer = get_normalizer(dataset_stats)
    pretrained_byola.to(device)

    if only_highlevel_features: 
        return lambda data: pretrained_byola(normalizer(data))[:, 1024:]
    return lambda data: pretrained_byola(normalizer(data))