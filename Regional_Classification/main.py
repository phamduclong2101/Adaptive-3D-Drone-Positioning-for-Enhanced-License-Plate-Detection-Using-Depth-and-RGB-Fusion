import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os
import json  # Thư viện để lưu lịch sử dưới dạng JSON
from data_loader import get_data_loaders
from CNNNet import DronePositionClassifier
from train import train_model

# Hàm chuẩn bị dữ liệu
def prepare_data(data_dir, batch_size):
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    return dataloaders, dataset_sizes

# Hàm khởi tạo mô hình
def initialize_model(num_classes, device):
    model = DronePositionClassifier(num_classes=num_classes)
    model = model.to(device)
    return model

# Hàm huấn luyện và lưu biểu đồ metric và lịch sử huấn luyện
def train_drone_model_with_metrics(data_dir, batch_size, num_classes, num_epochs, device, 
                                   model_save_dir, history_save_dir, metrics_save_dir):
    dataloaders, dataset_sizes = prepare_data(data_dir, batch_size)
    model = initialize_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, history = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=num_epochs, device=device)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(history_save_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)

    # Lưu mô hình
    model_file_path = os.path.join(model_save_dir, 'best_drone_model_2.pth')
    torch.save(model.state_dict(), model_file_path)
    print(f"Model saved to {model_file_path}")

    # Lưu lịch sử huấn luyện dưới dạng JSON
    history_file_path = os.path.join(history_save_dir, 'training_history_2.json')
    with open(history_file_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_file_path}")

    # Vẽ biểu đồ loss
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], 'b', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(metrics_save_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

    # Vẽ biểu đồ accuracy
    plt.figure()
    plt.plot(epochs, history['train_acc'], 'b', label='Training accuracy')
    plt.plot(epochs, history['val_acc'], 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_plot_path = os.path.join(metrics_save_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy plot saved to {accuracy_plot_path}")

    return model, dataloaders

# Hàm chính để thực thi toàn bộ quy trình
def main():
    data_dir = r'C:\Users\Hi Windows 11 Home\Documents\Multiclass_Classifier\positions_classification\Dataset_Split_2'
    num_classes = 4
    num_epochs = 25
    batch_size = 16

    # Các thư mục để lưu model, history và các biểu đồ metrics
    model_save_dir = r'C:\Users\Hi Windows 11 Home\Documents\Multiclass_Classifier\positions_classification\model_weights\model'
    history_save_dir = r'C:\Users\Hi Windows 11 Home\Documents\Multiclass_Classifier\positions_classification\model_weights\history'
    metrics_save_dir = r'C:\Users\Hi Windows 11 Home\Documents\Multiclass_Classifier\positions_classification\Chart_metric'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, dataloaders = train_drone_model_with_metrics(data_dir, batch_size, num_classes, num_epochs, 
                                                        device, model_save_dir, history_save_dir, metrics_save_dir)

if __name__ == "__main__":
    main()
