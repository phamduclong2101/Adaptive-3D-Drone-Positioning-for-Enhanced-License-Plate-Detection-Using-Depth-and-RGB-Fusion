import os
import torch
from torchvision import datasets, transforms

class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Dataset chỉ sử dụng ảnh từ thư mục image_dir.
        Nhãn sẽ được tự động gán dựa trên cấu trúc thư mục (dựa trên thư viện ImageFolder của PyTorch).

        :param image_dir: Đường dẫn đến thư mục chứa ảnh.
        :param transform: Phép biến đổi cho ảnh.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_dataset = datasets.ImageFolder(root=image_dir, transform=transform)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Lấy ảnh và nhãn từ ImageFolder
        image, label = self.image_dataset[idx]

        return image, label


def get_data_loaders(data_dir, batch_size=16):
    """
    Tạo DataLoader cho ảnh từ thư mục data_dir với cấu trúc thư mục:
    data_dir/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   ├── class_3/
    │   └── class_4/
    ├── val/
    └── test/

    :param data_dir: Đường dẫn đến thư mục chứa dữ liệu.
    :param batch_size: Kích thước batch.
    :return: Dataloader cho train, validation và test.
    """
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Sử dụng ImageFolder để tự động gán nhãn dựa trên cấu trúc thư mục
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=data_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=data_transform)

    # Tạo DataLoader cho train, validation và test
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
