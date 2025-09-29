import os.path
import cv2
import random
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor

class CrackDataset(BaseDataset):
    """A dataset class for crack dataset with dynamic 80/20 splitting."""

    def __init__(self, args):
        """Initialize this dataset class.

        This class now handles dynamic splitting of the dataset into training and
        validation sets based on a fixed random seed.
        """
        BaseDataset.__init__(self, args)

        # --- START: New data splitting logic ---

        # 1. Define common directories. We assume all images are in '.../img/'
        #    and all labels are in '.../lab/'.
        img_dir = os.path.join(args.dataset_path, 'img')
        self.lab_dir = os.path.join(args.dataset_path, 'lab')

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}. Please ensure your data is in 'data/crack500/img/'.")
        if not os.path.isdir(self.lab_dir):
            raise FileNotFoundError(f"Label directory not found: {self.lab_dir}. Please ensure your data is in 'data/crack500/lab/'.")

        # 2. Get all image paths and shuffle them reproducibly.
        all_img_paths = sorted(make_dataset(img_dir))
        random.Random(args.seed).shuffle(all_img_paths)

        # 3. Calculate the 80% split point.
        split_idx = int(len(all_img_paths) * 0.8)
        if len(all_img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}.")

        # 4. Assign the correct slice of paths based on the current phase.
        if args.phase == 'train':
            self.img_paths = all_img_paths[:split_idx]
            print(f"Dataset: Using {len(self.img_paths)} images for training (80% of total).")
        elif args.phase == 'test':
            self.img_paths = all_img_paths[split_idx:]
            print(f"Dataset: Using {len(self.img_paths)} images for validation (20% of total).")
        else:
            # Default behavior if phase is not train/test, e.g., for prediction.
            self.img_paths = all_img_paths

        # --- END: New data splitting logic ---

        # 使用 ImageNet 的标准均值和方差，以匹配 SAM/Hiera 预训练模型的要求
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.lab_transform = MaskToTensor()
        self.phase = args.phase

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        img_path = self.img_paths[index]
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        lab_path = os.path.join(self.lab_dir, base_filename + '.jpg')

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not os.path.exists(lab_path):
            # Try to find the corresponding .png file if .jpg is not found
            lab_path_png = os.path.join(self.lab_dir, base_filename + '.png')
            if os.path.exists(lab_path_png):
                lab_path = lab_path_png
            else:
                raise FileNotFoundError(f"Label file not found for image {img_path}. Looked for: {lab_path} and {lab_path_png}")

        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        w, h = self.args.load_width, self.args.load_height
        if w > 0 and h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)
        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
