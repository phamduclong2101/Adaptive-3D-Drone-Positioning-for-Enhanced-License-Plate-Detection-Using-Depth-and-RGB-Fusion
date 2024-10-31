from typing import Tuple
import torch
from torch import nn, optim
from CNN_Regression import CNNRegression
from data_loading import RegressionTaskData
import numpy as np

def train_network(device, epochs=100, image_size_rgb=(3, 100, 100), image_size_depth=(1, 100, 100)):
    model = CNNRegression(image_size_rgb=image_size_rgb, image_size_depth=image_size_depth).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Khởi tạo dữ liệu huấn luyện với cả ảnh RGB và Depth
    task_data = RegressionTaskData(image_size_rgb=image_size_rgb, image_size_depth=image_size_depth)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for rgb_inputs, depth_inputs, targets in task_data.trainloader:
            # Chuyển inputs và targets sang float32 và chuyển sang device
            rgb_inputs = rgb_inputs.to(device).float()
            depth_inputs = depth_inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(rgb_inputs, depth_inputs)  # Truyền cả RGB và Depth vào mô hình
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(task_data.trainloader):.4f}")

    return model

def save_model(model, filename='model.pth'):
    """
    Save the trained model to a file.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(image_size_rgb=(3, 100, 100), image_size_depth=(1, 100, 100), filename='model.pth'):
    """
    Load the saved model from a file.
    """
    model = CNNRegression(image_size_rgb=image_size_rgb, image_size_depth=image_size_depth)
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set model to evaluation mode
    return model


def evaluate_network(model, device, image_size_rgb: Tuple[int, int, int] = (3, 100, 100), image_size_depth: Tuple[int, int, int] = (1, 100, 100)):
    """
    Evaluate the model on the test dataset.
    """
    assert image_size_rgb[1] == image_size_rgb[2], 'RGB image size must be square'
    assert image_size_depth[1] == image_size_depth[2], 'Depth image size must be square'
    
    # Khởi tạo dữ liệu kiểm tra với cả ảnh RGB và Depth
    regression_task = RegressionTaskData(image_size_rgb=image_size_rgb, image_size_depth=image_size_depth)
    criterion = torch.nn.MSELoss()

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_loss = 0.0
        total_angle_error = 0.0
        n_samples_total = 0
        for rgb_inputs, depth_inputs, targets in regression_task.trainloader:
            rgb_inputs = rgb_inputs.to(device).float()
            depth_inputs = depth_inputs.to(device).float()
            targets = targets.to(device).float()

            outputs = model(rgb_inputs, depth_inputs)  # Truyền cả RGB và Depth vào mô hình
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Calculate angle error (assuming outputs[0], outputs[1] are x, y)
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            output_angles = np.arctan2(outputs_np[:, 1], outputs_np[:, 0])
            target_angles = np.arctan2(targets_np[:, 1], targets_np[:, 0])
            angle_error = np.sum(np.abs(np.rad2deg(target_angles - output_angles)))
            total_angle_error += angle_error
            n_samples_total += len(output_angles)

        mean_loss = total_loss / len(regression_task.valloader)
        mean_angle_error = total_angle_error / n_samples_total
        print(f'Test Loss: {mean_loss:.4f}')
        # print(f'Mean Angle Error: {mean_angle_error:.4f} degrees')
