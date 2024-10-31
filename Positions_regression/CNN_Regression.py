# import torch
# import torch.nn as nn
# from typing import Tuple

# class CNNRegression(nn.Module):
#     """
#     CNN model for regression task with dual backbone for RGB and Depth images.
#     """
#     def __init__(self, image_size_rgb: Tuple[int, int, int] = (3, 100, 100), image_size_depth: Tuple[int, int, int] = (1, 100, 100)):
#         super(CNNRegression, self).__init__()
        
#         # Backbone xử lý ảnh RGB
#         self.image_size_rgb = image_size_rgb
#         self.rgb_conv1 = nn.Conv2d(in_channels=self.image_size_rgb[0], out_channels=4, kernel_size=3, stride=1, padding=1)
#         self.rgb_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.rgb_conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.rgb_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Backbone xử lý ảnh Depth
#         self.image_size_depth = image_size_depth
#         self.depth_conv1 = nn.Conv2d(in_channels=self.image_size_depth[0], out_channels=4, kernel_size=3, stride=1, padding=1)
#         self.depth_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.depth_conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.depth_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Tính toán số lượng phần tử sau conv và pool layers cho cả RGB và Depth
#         self.fc1_input_features_rgb = 16 * 25 * 44  # Số lượng phần tử sau conv+pool của RGB
#         self.fc1_input_features_depth = 16 * 25 * 44  # Số lượng phần tử sau conv+pool của Depth
        
#         # Kết hợp đầu ra của cả hai backbone
#         self.fc1_rgb = nn.Linear(in_features=self.fc1_input_features_rgb, out_features=128)  # Fully connected cho chỉ RGB
#         self.fc1_combined = nn.Linear(in_features=self.fc1_input_features_rgb + self.fc1_input_features_depth, out_features=128)  # Fully connected cho cả RGB và Depth
#         self.fc2 = nn.Linear(in_features=128, out_features=3)

#     def forward(self, x_rgb, x_depth=None):
#         # Backbone xử lý ảnh RGB
#         x_rgb = self.rgb_conv1(x_rgb)
#         x_rgb = nn.functional.relu(x_rgb)
#         x_rgb = self.rgb_pool1(x_rgb)
#         x_rgb = self.rgb_conv2(x_rgb)
#         x_rgb = nn.functional.relu(x_rgb)
#         x_rgb = self.rgb_pool2(x_rgb)
        
#         # Flatten ảnh RGB
#         x_rgb = x_rgb.view(x_rgb.size(0), -1)

#         if x_depth is not None:
#             # Backbone xử lý ảnh Depth (chỉ sử dụng trong quá trình huấn luyện)
#             x_depth = self.depth_conv1(x_depth)
#             x_depth = nn.functional.relu(x_depth)
#             x_depth = self.depth_pool1(x_depth)
#             x_depth = self.depth_conv2(x_depth)
#             x_depth = nn.functional.relu(x_depth)
#             x_depth = self.depth_pool2(x_depth)

#             # Flatten ảnh Depth
#             x_depth = x_depth.view(x_depth.size(0), -1)

#             # Kết hợp đầu ra của RGB và Depth bằng cách concatenate
#             x = torch.cat((x_rgb, x_depth), dim=1)

#             # Qua lớp fully connected cho cả RGB và Depth
#             x = self.fc1_combined(x)
#         else:
#             # Khi inference chỉ có RGB, sử dụng đầu ra của RGB
#             x = self.fc1_rgb(x_rgb)

#         x = nn.functional.relu(x)
#         x = self.fc2(x)
        
#         return x




import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple

class CNNRegression(nn.Module):
    """
    ResNet model for regression task with dual backbone for RGB and Depth images.
    """
    def __init__(self, image_size_rgb: Tuple[int, int, int] = (3, 224, 224), image_size_depth: Tuple[int, int, int] = (1, 224, 224)):
        super(CNNRegression, self).__init__()
        
        # Backbone xử lý ảnh RGB (sử dụng ResNet)
        self.rgb_resnet = models.resnet18(pretrained=True)  # Sử dụng ResNet18
        self.rgb_resnet.fc = nn.Identity()  # Loại bỏ fully connected layer cuối cùng

        # Backbone xử lý ảnh Depth (ResNet phiên bản grayscale)
        self.depth_resnet = models.resnet18(pretrained=False)
        self.depth_resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_resnet.fc = nn.Identity()  # Loại bỏ fully connected layer cuối cùng

        # Fully connected layers
        self.fc1_rgb = nn.Linear(in_features=512, out_features=128)  # Fully connected cho chỉ RGB
        self.fc1_combined = nn.Linear(in_features=512 * 2, out_features=128)  # Fully connected cho cả RGB và Depth
        self.fc2 = nn.Linear(in_features=128, out_features=3)

    def forward(self, x_rgb, x_depth=None):
        # Forward pass cho ảnh RGB
        x_rgb = self.rgb_resnet(x_rgb)

        if x_depth is not None:
            # Forward pass cho ảnh Depth
            x_depth = self.depth_resnet(x_depth)

            # Kết hợp đầu ra của RGB và Depth
            x = torch.cat((x_rgb, x_depth), dim=1)

            # Qua lớp fully connected cho cả RGB và Depth
            x = self.fc1_combined(x)
        else:
            # Khi inference chỉ có RGB, sử dụng đầu ra của RGB
            x = self.fc1_rgb(x_rgb)

        x = nn.functional.relu(x)
        x = self.fc2(x)
        
        return x
