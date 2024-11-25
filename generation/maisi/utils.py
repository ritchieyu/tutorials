import torch
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)

from typing import Dict

from pvg.constants.loris.general import Sequences
from pvg.runner.transforms.augmentation import CustomTransforms

# class Compose:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, data_dict):
#         for i, transform in enumerate(self.transforms):
#             data_dict = transform(data_dict)
#         return data_dict

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

from typing import List, Optional

def define_train_transform(
    random_aug: bool,
    patch_size: List[int] = [128, 128, 128],
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["MRI"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
) -> tuple:
    
    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    # ------------------------
    # Define common transforms
    # ------------------------
    common_transform = []  # NOTE: ignoring SelectItemsd, LoadImaged, EnsureChannelFirstd, Orientationd, which are implemented in the original MAISI transforms pipeline

    common_transform.append(Lambdad(keys=image_keys, func=lambda x: x[select_channel : select_channel + 1, ...]))
    common_transform.append(ScaleIntensityRangePercentilesd(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False))  # fixed intensity transformation

    if spacing_type == "fixed":
        common_transform.append(
            Spacingd(keys=image_keys + label_keys, allow_missing_keys=True, pixdim=spacing, mode=interp_mode)
        )
    
    # ------------------------
    # Define random transforms
    # ------------------------
    random_transform = []

    if random_aug:

        # intensity transforms
        random_transform.append(RandBiasFieldd(keys=image_keys, prob=0.3, coeff_range=(0.0, 0.3)))
        random_transform.append(RandGibbsNoised(keys=image_keys, prob=0.3, alpha=(0.5, 1.0)))
        random_transform.append(RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.5, 2.0)))
        random_transform.append(RandHistogramShiftd(keys=image_keys, prob=0.05, num_control_points=10))

        # spatial transforms
        for axis in range(3):
            random_transform.append(RandFlipd(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axis=axis))

        for axes in [(0, 1), (1, 2), (0, 2)]:
            random_transform.append(RandRotate90d(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axes=axes))

        random_transform.append(RandScaleIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, factors=(0.9, 1.1)))
        random_transform.append(RandShiftIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, offsets=0.05))

        if spacing_type == 'rand_zoom':
            random_transform.append(RandZoomd(keys=image_keys + label_keys, allow_missing_keys=True, prob=0.3, min_zoom=0.5, max_zoom=1.5, keep_size=False, mode=interp_mode))
            random_transform.append(RandRotated(keys=image_keys + label_keys, allow_missing_keys=True, prob=0.3, range_x=0.1, range_y=0.1, range_z=0.1, keep_size=True, mode=interp_mode))

    # -------------------
    # Cropping transforms
    # -------------------
    crop_transform = []
    crop_transform.append(SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True))
    crop_transform.append(RandSpatialCropd(keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True))

    # ----------------
    # Final transforms
    # ----------------
    final_transform = []
    final_transform.append(EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True))

    # ----------------------
    # Compose all transforms
    # ----------------------
    if random_aug:
        train_transforms = Compose(common_transform + random_transform + crop_transform + final_transform)
    else:
        train_transforms = Compose(common_transform + crop_transform + final_transform)

    return train_transforms


def define_val_transform(
    k: int = 4,
    val_patch_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["image"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
) -> tuple:
    
    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    # ------------------------
    # Define common transforms
    # ------------------------
    common_transform = []  # NOTE: ignoring SelectItemsd, LoadImaged, EnsureChannelFirstd, Orientationd, which are implemented in the original MAISI transforms pipeline

    common_transform.append(Lambdad(keys=image_keys, func=lambda x: x[select_channel : select_channel + 1, ...]))
    common_transform.append(ScaleIntensityRangePercentilesd(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False))  # fixed intensity transformation

    if spacing_type == "fixed":
        common_transform.append(
            Spacingd(keys=image_keys + label_keys, allow_missing_keys=True, pixdim=spacing, mode=interp_mode)
        )

    # ----------------------
    # Define crop transforms
    # ----------------------
    crop_transform = []

    if val_patch_size is None:
        crop_transform.append(DivisiblePadd(keys=keys, allow_missing_keys=True, k=k))
    else:
        crop_transform.append(ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size))
    
    # ----------------
    # Final transforms
    # ----------------
    final_transform = []
    final_transform.append(EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True))

    # ----------------------
    # Compose all transforms
    # ----------------------
    val_transforms = Compose(tuple(common_transform + crop_transform + final_transform))

    return val_transforms


class MAISI_Transform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img: dict) -> dict:
        '''
        Apply transformation pipeline to input image.
        '''
        return self.transform(img)