"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, args).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
import os
import glob
import random
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from datasets.base_dataset import BaseDataset
from datasets.crack_tree_dataset import crack_tree_dataset


def find_dataset_using_name(dataset_name):
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'

    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(args):
    """Create a dataset given the option."""
    data_loader = CustomDatasetDataLoader(args)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, args):
        """Initialize this class"""
        self.args = args

        if 'CrackTree260' in args.dataset_path:
            img_dir = os.path.join(args.dataset_path, 'img')
            lab_dir = os.path.join(args.dataset_path, 'lab')
            all_image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
            all_mask_files = sorted([os.path.join(lab_dir, f) for f in os.listdir(lab_dir) if f.lower().endswith('.bmp')])

            if len(all_image_files) != len(all_mask_files):
                raise ValueError(f"图像和掩码文件的数量不匹配 (图像: {len(all_image_files)} vs 掩码: {len(all_mask_files)})。")

            indices = list(range(len(all_image_files)))
            random.Random(args.seed).shuffle(indices)
            
            split_ratio = 0.8
            split_point = int(len(indices) * split_ratio)
            
            train_indices = indices[:split_point]
            val_indices = indices[split_point:]

            if args.phase == 'train':
                selected_indices = train_indices
                # --- [FIXED] Use RandomCrop for training ---
                transform = A.Compose([
                    A.RandomCrop(height=args.load_height, width=args.load_width, always_apply=True),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            else: # 'test' phase for validation
                selected_indices = val_indices
                transform = A.Compose([
                    A.CenterCrop(height=args.load_height, width=args.load_width, always_apply=True),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])

            image_paths = [all_image_files[i] for i in selected_indices]
            mask_paths = [all_mask_files[i] for i in selected_indices]

            if not image_paths and args.phase == 'test':
                 print("警告：验证集为空。这可能是因为数据集太小或划分比例不合适。")
            elif not image_paths:
                 raise ValueError(f"错误：在为 '{args.phase}' 阶段划分数据集后，没有剩余的样本。")

            # --- [FIXED] Removed unexpected keyword arguments ---
            self.dataset = crack_tree_dataset(
                image_paths=image_paths, 
                mask_paths=mask_paths, 
                transform=transform
            )
            
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=args.batch_size,
                shuffle=not args.serial_batches and args.phase == 'train',
                num_workers=int(args.num_threads)
            )
            return

        # --- Original logic for all other datasets ---
        dataset_class = find_dataset_using_name(args.dataset_mode)
        self.dataset = dataset_class(args)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=not args.serial_batches,
            num_workers=int(args.num_threads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
