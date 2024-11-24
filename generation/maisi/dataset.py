from monai import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
from typing import Union, List, Tuple

from pvg.constants.loris.general import Sequences
from pvg.constants.loris.indexing import SubjectKeys as SJ_KEYS
from pvg.constants.pipeline import SplitKeys
from pvg.runner.dataset.dataset import get_leaf_val_from_dict, glob_file, load_image

from utils import Hyperparameters, Compose


# TODO: Modify the transforms used for training

class DatasetDescription:
    """
        Creates the train/val/test datasets given a data split and a dict containing all
        the required rootdirs to find a file.

        If the train/val/test dataset is a Dataset object, the default DataLoader
        will be used with Hyperparameters.batch_size and Hyperparameters.num_workers.

        If the train/val/test dataset is a list/tuple of Dataset objects, the custom
        DataLoader will be used instead.
    """
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup_datasets(self, data_split, rootdir_dict):
        ################## MODIFY HERE #######################
        item_template = {
                'MRI': [(SJ_KEYS.MRI_AND_LABEL, s) for s in Hyperparameters.sequences],
                'MASK': [(SJ_KEYS.MRI_AND_LABEL, Sequences.BEAST)],
        }
        self.train_dataset = CustomVolumeLoader(data_split[SplitKeys.TRAIN], rootdir_dict, item_template,
                transform=Compose([
                    transforms.SpatialPadd(**Hyperparameters.transforms['model']['SpatialPadd']),
                    # transforms.RandFlipd(**Hyperparameters.transforms['model']['RandFlipd']),  # TODO: Do all transforms get applied to the mask as well?
                    ]))
        self.val_dataset = CustomVolumeLoader(data_split[SplitKeys.VALIDATION], rootdir_dict, item_template,
                transform=Compose([
                    transforms.SpatialPadd(**Hyperparameters.transforms['model']['SpatialPadd'])
                    ]))
        self.test_dataset = CustomVolumeLoader(data_split[SplitKeys.TEST], rootdir_dict, item_template,
                transform=Compose([
                    transforms.SpatialPadd(**Hyperparameters.transforms['model']['SpatialPadd'])
                    ]))
        ######################################################

    def get_train_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return [self.train_dataset]

    def get_val_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return [self.val_dataset]

    def get_test_dataset(self) -> Union[Dataset, List[Dataset], Tuple[Dataset]]:
        return [self.test_dataset]
    
class CustomVolumeLoader(Dataset):
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

    @staticmethod
    def get_train_loader(dataset):
        dataloader = DataLoader( 
            dataset,
            batch_size=Hyperparameters.batch_size,
            drop_last=True,
            num_workers=Hyperparameters.num_workers,
            pin_memory=True,
            persistent_workers=True)
        return dataloader

    @staticmethod
    def get_val_loader(dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=Hyperparameters.batch_size,
            drop_last=True,
            num_workers=Hyperparameters.num_workers,
            pin_memory=True,
            persistent_workers=True)
        return dataloader

    @staticmethod
    def get_test_loader(dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=Hyperparameters.batch_size,
            drop_last=True,
            num_workers=Hyperparameters.num_workers,
            pin_memory=True)
        return dataloader