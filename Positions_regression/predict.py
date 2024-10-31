# import torch
# from torchvision import transforms
# from PIL import Image
# from CNN_Regression import CNNRegression
# from typing import Tuple
# import numpy as np

# def predict_image_rgb(model, device, rgb_image_path: str, 
#                       image_size_rgb: Tuple[int, int, int] = (3, 100, 100)):
#     """
#     Predict output for a single RGB image (no Depth).
#     """
#     # Transforms cho ảnh RGB
#     transform_rgb = transforms.Compose([
#         transforms.Resize(image_size_rgb[1]),  # Resize để phù hợp với kích thước đầu vào của mô hình
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize RGB images
#     ])

#     # Mở và transform ảnh RGB
#     rgb_image = Image.open(rgb_image_path).convert('RGB')
#     rgb_input_tensor = transform_rgb(rgb_image).unsqueeze(0)  # Thêm batch dimension
#     rgb_input_tensor = rgb_input_tensor.to(device)  # Chuyển tensor sang device

#     with torch.no_grad():
#         # Dự đoán chỉ dựa trên ảnh RGB
#         output = model(rgb_input_tensor, None)  # Truyền None cho Depth vì không sử dụng Depth
#         return output.cpu().numpy().flatten()


# # Adjusted angle calculation for 3D outputs (ignoring Z for now)
# def calculate_angle_error(outputs_np, targets_np):
#     """
#     Calculate angle error based on the predicted and target x, y coordinates.
#     Z coordinate is ignored for the angle calculation.
#     """
#     # Assuming Z does not affect the angle calculation for simplicity
#     output_angles = np.array([np.arctan2(out[1], out[0]) for out in outputs_np])  # Adjusted for (x, y) axis
#     target_angles = np.array([np.arctan2(t[1], t[0]) for t in targets_np])  # Adjusted for (x, y) axis
#     angle_error = np.sum(np.abs(np.rad2deg(target_angles - output_angles)))
#     return angle_error



import torch
from torchvision import transforms
from PIL import Image
from CNN_Regression import CNNRegression
from typing import Tuple
import numpy as np

def predict_image(model, device, rgb_image_path: str, depth_image_path: str,
                  image_size_rgb: Tuple[int, int, int] = (3, 100, 100),
                  image_size_depth: Tuple[int, int, int] = (1, 100, 100)):
    """
    Predict output for a single pair of RGB and Depth images.
    """
    # Transforms for RGB and Depth images
    transform_rgb = transforms.Compose([
        transforms.Resize(image_size_rgb[1]),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize RGB images
    ])

    transform_depth = transforms.Compose([
        transforms.Resize(image_size_depth[1]),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize depth images
    ])

    # Open and transform the RGB image
    rgb_image = Image.open(rgb_image_path).convert('RGB')
    rgb_input_tensor = transform_rgb(rgb_image).unsqueeze(0)  # Add batch dimension
    rgb_input_tensor = rgb_input_tensor.to(device)  # Move tensor to device

    # Open and transform the Depth image
    depth_image = Image.open(depth_image_path).convert('L')  # Depth image is grayscale
    depth_input_tensor = transform_depth(depth_image).unsqueeze(0)  # Add batch dimension
    depth_input_tensor = depth_input_tensor.to(device)  # Move tensor to device

    with torch.no_grad():
        # Predict using both RGB and Depth images
        output = model(rgb_input_tensor, depth_input_tensor)
        return output.cpu().numpy().flatten()


# Adjusted angle calculation for 3D outputs (ignoring Z for now)
def calculate_angle_error(outputs_np, targets_np):
    """
    Calculate angle error based on the predicted and target x, y coordinates.
    Z coordinate is ignored for the angle calculation.
    """
    # Assuming Z does not affect the angle calculation for simplicity
    output_angles = np.array([np.arctan2(out[1], out[0]) for out in outputs_np])  # Adjusted for (x, y) axis
    target_angles = np.array([np.arctan2(t[1], t[0]) for t in targets_np])  # Adjusted for (x, y) axis
    angle_error = np.sum(np.abs(np.rad2deg(target_angles - output_angles)))
    return angle_error
