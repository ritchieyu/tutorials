import torch
from typing import Dict

from pvg.constants.loris.general import Sequences
from pvg.runner.transforms.augmentation import CustomTransforms

class Hyperparameters:
    batch_size: int = 4
    num_workers: int = 7
    sequences = [Sequences.FLR]
    transforms: Dict[str, Dict] = {
            'model': {
                'SpatialPadd': {
                    'keys': ['MRI', 'MASK'],
                    'spatial_size': [64, 80, 64]
                    # 'spatial_size': [64, 256, 256]
                },
                'RandFlipd': {
                    'keys': ['MRI', 'MASK'],
                    'prob': 0
                }
            },
        }

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict):
        for i, transform in enumerate(self.transforms):
            data_dict = transform(data_dict)
        return data_dict

def build_batch(batch, device):
    for key, value in batch.items():
        if key in ['MRI', 'MASK']:
            batch[key] = value.to(device, non_blocking=True)

    seq = batch['MRI']
    mask = batch['MASK']
    
    # preprocess data
    seq = seq.mul(mask)
    seq[mask==1] = seq[mask==1] - torch.amin(seq[mask==1])

    # ... get 99.5 intensity percentile inside mask...
    seq[mask==1] = seq[mask==1] / torch.quantile(seq[mask==1], 0.995)  # TODO: Clip between 0 and 1

    # ... clip values between 0 and 1
    seq = torch.clamp(seq, 0, 1)

    batch['MRI'] = seq
    batch['MASK'] = mask

    return batch