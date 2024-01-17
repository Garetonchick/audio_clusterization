import os
import torch
import torchaudio
from tqdm import tqdm
import nnAudio.features
from byol_a import get_byola_cfg

class WavToLogMelSpectogramDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, audio_files, labels, audio_dir='', cache=True):
        self.audio_files = audio_files
        self.audio_dir = audio_dir
        self.labels = labels
        self.to_melspec = nnAudio.features.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        )
        self.cached_spectograms = [None] * len(audio_files) if cache else None
        self.cache = cache
        self.sample_rate = cfg.sample_rate

    def __len__(self):
        return len(self.audio_files)

    def calc_norm_stats(self):
        X = torch.tensor([])
        lmss = []
        for lms, _ in tqdm(self, desc="calc_norm_stats"):
            lmss.append(lms)
        X = torch.cat(lmss, dim=1)
        return [X.mean().item(), X.std().item()]

    def __getitem__(self, idx):
        if self.cache and self.cached_spectograms[idx] is not None:
            return self.cached_spectograms[idx], self.labels[idx]

        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        wav, sr = torchaudio.load(audio_path)
        assert sr == self.sample_rate, "Wrong audio sample rate"
        wav = wav.mean(dim=0, keepdim=True) # Reduce wav to one channel
        lms = (self.to_melspec(wav) + torch.finfo(torch.float).eps).log()

        if self.cache:
            self.cached_spectograms[idx] = lms

        return lms, self.labels[idx]

def get_audio_files_list(dir):
    audio_files = sorted(os.listdir(os.path.join(dir, 'audio')))
    return audio_files

def get_labels(dir):
    audio_files = get_audio_files_list(dir)
    labels_dict = {}
    with open(os.path.join(dir, 'labels.txt')) as f:
        labels_dict = {tup[0][6:] : tup[1] for tup in [line.split('\t') for line in f.readlines()]}
    return [labels_dict[filename] for filename in audio_files]

def get_dataset(dir=''):
    return WavToLogMelSpectogramDataset(
        cfg=get_byola_cfg(),
        audio_files=get_audio_files_list(dir),
        labels=get_labels(dir),
        audio_dir=os.path.join(dir, 'audio'))