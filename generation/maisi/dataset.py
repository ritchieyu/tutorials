import torch
from torch.utils.data import Dataset, DataLoader
from monai import transforms
from monai.transforms import convert_to_contiguous

import os
import numpy as np
from typing import Union, List, Tuple

from pvg.constants.loris.general import Sequences
from pvg.constants.loris.indexing import SubjectKeys as SJ_KEYS
from pvg.constants.pipeline import SplitKeys
from pvg.runner.dataset.dataset import get_leaf_val_from_dict, glob_file, load_image

from utils import MAISI_Transform


# TODO: Modify the transforms used for training

class DatasetDescription:
    """
        Creates the train/val/test datasets given a data split and a dict containing all
        the required rootdirs to find a file.

        If the train/val/test dataset is a Dataset object, the default DataLoader
        will be used with Hyperparameters.batch_size and Hyperparameters.num_workers.

        If the train/val/test dataset is a list/tuple of Dataset objects, the custom
        DataLoader will be used instead.

        CHANGES:
        - Transform must be defined outside of this class.
    """
    def __init__(self, sequences):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.sequences = sequences

    def setup_datasets(self, data_split, rootdir_dict, train_transform, val_transform, test_transform):
        ################## MODIFY HERE #######################
        item_template = {
                'MRI': [(SJ_KEYS.MRI_AND_LABEL, s) for s in self.sequences],
                'MASK': [(SJ_KEYS.MRI_AND_LABEL, Sequences.BEAST)],
        }
        self.train_dataset = CustomVolumeLoader(data_split[SplitKeys.TRAIN], rootdir_dict, item_template,
                transform=train_transform)
        self.val_dataset = CustomVolumeLoader(data_split[SplitKeys.VALIDATION], rootdir_dict, item_template,
                transform=val_transform)
        self.test_dataset = CustomVolumeLoader(data_split[SplitKeys.TEST], rootdir_dict, item_template,
                transform=test_transform)
        ######################################################

    def get_train_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return [self.train_dataset]

    def get_val_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return [self.val_dataset]

    def get_test_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return [self.test_dataset]
    
class CustomVolumeLoader(Dataset):  # dataset class
    def __init__(self, dataset_dict, rootdir_dict, item_template, transform=None):
        """
            dataset_dict: dict with format of { sample_key: sample_dict }
            rootdir_dict: dict with format of { sample_dict_key: rootdir }
            item_template: template dict for the item returned by the dataset.
                Dict has format of { item_key1: [ keys_for_ch1, keys_for_ch2, ... ], ... }
                where keys_for_ch is a tuple of keys to access the filepath of an image to load

            ASSUMPTIONS:
            - keys in each sample_dict corresponds to a key in rootdir_dict
            - the first key in each keys_for_ch corresponds to a key in rootdir_dict
        """
        self.dataset_dict = dataset_dict
        self.sample_keys = list(dataset_dict.keys())
        self.rootdir_dict = rootdir_dict
        self.item_template = item_template
        self.transform = transform

    def __len__(self):
        return len(self.sample_keys)

    def _load_vol(self, idx):
        sample_key = self.sample_keys[idx]
        sample_dict = self.dataset_dict[sample_key]

        # sample random timepoint from subject
        random_tmpt = np.random.choice(list(sample_dict[SJ_KEYS.MRI_AND_LABEL]))

        # Load images
        output = {}
        for item_key, keys_for_ch_list in self.item_template.items():
            # Get image paths
            img_paths = []
            for keys_for_ch in keys_for_ch_list:
                assert len(keys_for_ch) == 2
                keys_for_ch = [keys_for_ch[0]] + [random_tmpt] + [keys_for_ch[-1]]
                rel_path = get_leaf_val_from_dict(sample_dict, keys_for_ch)
                full_path = os.path.join(self.rootdir_dict[keys_for_ch[0]], rel_path)
                full_path = glob_file(full_path)
                img_paths.append(full_path)

            # Load images and stack into one array
            imgs = [load_image(path) for path in img_paths]
            output[item_key] = np.stack(imgs)

        # Perform skull-stripping
        output["MRI"] = output["MRI"] * output["MASK"]
        output["MRI"] = torch.from_numpy(output["MRI"]).contiguous()  # TODO: Why does this not fix the error message?
        del output["MASK"]  # not needed anymore
        
        # output = convert_to_contiguous(output, memory_format=torch.contiguous_format)  # memory_format argument applies if input is a torch tensor

        # Run transformation pipeline
        if self.transform is not None:
            output = self.transform(output)

        return output

    def __getitem__(self, idx):
        return self._load_vol(idx)
    

class CustomDataLoader:
    """
        Custom DataLoader should either be a PyTorch DataLoader object or a custom python
        object with __len__ and __iter__ method implemented.
        If a custom object, __len__ should return the number of iterations/batches in an
        epoch, and __iter__ should return a new Iterator.

        CHANGES PERFORMED:
        - modified distributed loader to classic loader
        - changed 'dataset[0]' to 'dataset'
    """
    def __init__(self, train_batch_size, val_batch_size, test_batch_size, num_workers):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def get_train_loader(self, dataset):
        dataloader = DataLoader( 
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True)
        return dataloader

    def get_val_loader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True)
        return dataloader

    def get_test_loader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.test_batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True)
        return dataloader
    
if __name__ == '__main__':
    import torch
    import json
    import argparse
    import matplotlib.pyplot as plt
    from pvg.runner.utilities.io import IO
    from utils import define_train_transform, define_val_transform

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Parent directory of LORIS dataset (SLURM_TMPDIR)')
    pargs = parser.parse_args()

    data_dir = pargs.data_dir

    args = argparse.Namespace()

    # Load environment
    environment_file = "./configs/environment_maisi_vae_train.json"
    env_dict = json.load(open(environment_file, "r"))
    for k, v in env_dict.items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    # Load configuration files
    config_file = "./configs/config_maisi.json"
    config_dict = json.load(open(config_file, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)

    config_train_file = "./configs/config_maisi_vae_train.json"
    config_train_dict = json.load(open(config_train_file, "r"))
    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    
    train_transform = define_train_transform(random_aug=args.random_aug,
                                             patch_size=args.patch_size,
                                             output_dtype=torch.float16,  # final data type
                                             spacing_type=args.spacing_type,
                                             spacing=args.spacing,
                                             image_keys=["MRI"],
                                             label_keys=[],
                                             additional_keys=[],
                                             select_channel=0)

    
    val_transform = define_val_transform(k=4,  # patches should be divisible by k
                                         val_patch_size=args.val_patch_size,  # if None, will validate on whole image volume
                                         output_dtype=torch.float16,  # final data type
                                         image_keys=["MRI"],
                                         label_keys=[],
                                         additional_keys=[],
                                         select_channel=0)
    
    test_transform = val_transform

    splits_path = '/home/ritchiey/projects/def-arbeltal/ritchiey/capstone/old_splits.pickle'  # using 1x1x3_v4_0_w000 for now...
    rootdir_dict = {"MRI_AND_LABEL": f"{data_dir}/loris_1x1x3_v4_0_w000"}
    splits = IO.load_pickle(splits_path)
    dataset_descriptor = DatasetDescription(sequences=[Sequences.FLR])
    dataset_descriptor.setup_datasets(data_split=splits["experiment"],
                                      rootdir_dict=rootdir_dict,
                                      train_transform=MAISI_Transform(train_transform),
                                      val_transform=MAISI_Transform(val_transform),
                                      test_transform=MAISI_Transform(test_transform))

    dataloader = CustomDataLoader(train_batch_size=32,
                              val_batch_size=1,
                              test_batch_size=1,
                              num_workers=7)
    
    dataloader_train = dataloader.get_train_loader(dataset_descriptor.train_dataset)
    dataloader_val = dataloader.get_val_loader(dataset_descriptor.val_dataset)
    dataloader_test = dataloader.get_test_loader(dataset_descriptor.test_dataset)

    # for _ in dataloader_train:
        # continue

    sample = next(iter(dataloader_val))
    print(sample["MRI"].shape)

    # fig, ax = plt.subplots()
    # ax.imshow(sample['MRI'][0, 0, 32, :, :], cmap='gray')
    # fig.savefig("sample_transformed.png", dpi=500)