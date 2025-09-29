import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor

class CrackDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, args):
        """Initialize this dataset class.

        Parameters:
            args (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, args)
        self.img_paths = make_dataset(os.path.join(args.dataset_path, '{}_img'.format(args.phase)))
        self.lab_dir = os.path.join(args.dataset_path, '{}_lab'.format(args.phase))
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()

        self.phase = args.phase

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            image (tensor) - - an image
            label (tensor) - - its corresponding segmentation
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]

        # --- START: 修改区域 ---

        # 1. 从图像路径中分离出文件名（不含后缀）
        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # 2. 使用这个不含后缀的文件名来拼接标签路径，并确保标签后缀是.png
        lab_path = os.path.join(self.lab_dir, base_filename + '.png')

        # --- END: 修改区域 ---

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # 如果图像是灰度图，确保它被转换为3通道BGR图像
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 增加一个检查，确保标签文件存在
        if not os.path.exists(lab_path):
            raise FileNotFoundError(f"标签文件未找到！请为图片 {img_path} 检查对应的标签文件路径: {lab_path}")

        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # adjust the image size
        w, h = self.args.load_width, self.args.load_height
        # 确保尺寸不为0
        if w > 0 and h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        # 将标签转换为 0 和 1
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)
        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)