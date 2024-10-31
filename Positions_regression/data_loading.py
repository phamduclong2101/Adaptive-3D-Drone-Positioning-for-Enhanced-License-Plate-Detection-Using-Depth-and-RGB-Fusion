from typing import Dict, Any
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
from PIL import Image


def load_image_targets_from_csv(csv_path: Path, header: bool = True) -> Dict[str, Any]:
    """
    Load the image paths (RGB and Depth) and targets from a CSV file.
    The first column contains the RGB image path, the second column contains the Depth image path,
    and the subsequent columns contain the target values (x, y, z).
    """
    image_targets = {}

    # Check if CSV file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open('r') as f:
        lines = f.readlines()
        start_line = 0
        if header:
            start_line = 1

        for line in lines[start_line:]:
            values = line.strip().split(',')
            rgb_image_path = values[0]
            depth_image_path = values[1]
            targets = np.array([float(v) for v in values[2:5]])  # Get x, y, z
            image_targets[(rgb_image_path, depth_image_path)] = targets

    return image_targets


class RGBDepthDataset(Dataset):
    """
    Custom dataset for regression task, handling both RGB and Depth images with corresponding targets.
    """
    def __init__(self, image_targets: Dict[str, Any], transform_rgb=None, transform_depth=None) -> None:
        self.rgb_paths, self.depth_paths = zip(*image_targets.keys())  # Tách đường dẫn RGB và Depth
        self.targets = list(image_targets.values())  # Tọa độ mục tiêu (x, y, z)
        self.transform_rgb = transform_rgb  # Biến đổi cho ảnh RGB
        self.transform_depth = transform_depth  # Biến đổi cho ảnh Depth

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index: int):
        # Load RGB image
        rgb_image = Image.open(self.rgb_paths[index]).convert('RGB')
        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)

        # Load Depth image
        depth_image = Image.open(self.depth_paths[index]).convert('L')  # Depth image is grayscale
        if self.transform_depth:
            depth_image = self.transform_depth(depth_image)

        # Get target (x, y, z)
        target = torch.tensor(self.targets[index], dtype=torch.float32)

        return rgb_image, depth_image, target


class RegressionTaskData:
    """
    Wrapper class for the data used in the regression task. It contains train and test loaders for RGB and Depth images.
    """
    def __init__(self, image_size_rgb: Tuple[int, int, int] = (3, 100, 100), image_size_depth: Tuple[int, int, int] = (1, 100, 100), 
                 image_folder_path: Path = Path('Dataset_Merged/'), batch_size: int = 64) -> None:
        self.image_size_rgb = image_size_rgb
        self.image_size_depth = image_size_depth
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        
        # Ensure the dataset paths exist
        self.check_dataset_path(self.image_folder_path)

        # Define transforms for RGB and Depth images
        self.train_transforms_rgb, self.val_transforms_rgb = self.get_rgb_transforms(self.image_size_rgb[1])
        self.train_transforms_depth, self.val_transforms_depth = self.get_depth_transforms(self.image_size_depth[1])

        self.trainloader = self.make_loader('train.csv', 'train')
        self.valloader = self.make_loader('val.csv', 'val')
        self.testloader = self.make_loader('test.csv', 'test')

    def get_rgb_transforms(self, resize_size: int):
        """
        Return appropriate transforms for RGB images.
        """
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize RGB images
        ])
        return transform, transform

    def get_depth_transforms(self, resize_size: int):
        """
        Return appropriate transforms for Depth images.
        """
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize depth images
        ])
        return transform, transform

    def make_loader(self, csv_file: str, subset: str) -> torch.utils.data.DataLoader:
        """
        Build the DataLoader for the given subset (train, val, test).
        """
        image_targets = load_image_targets_from_csv(self.image_folder_path / csv_file)
        
        dataset = RGBDepthDataset(
            image_targets=image_targets,
            transform_rgb=self.train_transforms_rgb if subset == 'train' else self.val_transforms_rgb,
            transform_depth=self.train_transforms_depth if subset == 'train' else self.val_transforms_depth
        )

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=(subset == 'train'))

    def check_dataset_path(self, path: Path):
        """
        Ensure the dataset directory and necessary files exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {path}")
        for csv_file in ['train.csv', 'val.csv', 'test.csv']:
            if not (path / csv_file).exists():
                raise FileNotFoundError(f"{csv_file} not found in: {path}")
