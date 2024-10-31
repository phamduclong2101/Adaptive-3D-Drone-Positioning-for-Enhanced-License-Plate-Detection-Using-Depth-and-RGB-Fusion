import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class DronePositionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(DronePositionClassifier, self).__init__()
        
        # Mạng CNN sử dụng ResNet18 đã pretrained trên ImageNet
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Hoặc weights=ResNet18_Weights.DEFAULT
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # Xóa lớp fully connected để chỉ lấy đặc trưng

        # Khi không sử dụng tọa độ, chỉ cần fully connected cho ảnh
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image):
        # Trích xuất đặc trưng từ CNN cho ảnh
        cnn_features = self.cnn(image)
        
        # Đưa vào lớp fully connected để phân loại
        output = self.fc(cnn_features)
        
        return output
