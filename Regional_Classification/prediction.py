import os
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from CNNNet import DronePositionClassifier

def load_model(model_path, num_classes, device):
    """
    Load mô hình đã huấn luyện từ file .pth
    
    :param model_path: Đường dẫn tới file .pth chứa trọng số mô hình.
    :param num_classes: Số lượng class.
    :param device: Thiết bị (CPU/GPU) để chạy dự đoán.
    :return: Mô hình đã được load.
    """
    model = DronePositionClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Đặt mô hình vào chế độ dự đoán
    return model

def predict_image(image_path, model, device):
    """
    Hàm để dự đoán class của một ảnh.
    
    :param image_path: Đường dẫn tới ảnh cần dự đoán.
    :param model: Mô hình đã được huấn luyện.
    :param device: Thiết bị (CPU/GPU).
    :return: Nhãn dự đoán của ảnh.
    """
    # Chuẩn bị phép biến đổi giống với lúc huấn luyện
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Đọc và biến đổi ảnh
    image = Image.open(image_path).convert('RGB')
    transformed_image = data_transform(image)
    transformed_image = transformed_image.unsqueeze(0)  # Thêm batch size = 1

    # Đưa ảnh vào thiết bị
    transformed_image = transformed_image.to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(transformed_image)
        _, predicted = torch.max(output, 1)  # Lấy nhãn có xác suất cao nhất

    return predicted.item(), image  # Trả về nhãn và ảnh gốc để hiển thị

def select_random_image(test_dir):
    """
    Hàm để chọn ngẫu nhiên một ảnh từ thư mục test.
    
    :param test_dir: Thư mục chứa ảnh test.
    :return: Đường dẫn tới ảnh được chọn ngẫu nhiên.
    """
    # Lấy tất cả các đường dẫn tới các ảnh trong thư mục test
    all_images = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                all_images.append(os.path.join(root, file))
    
    # Chọn ngẫu nhiên một ảnh
    random_image = random.choice(all_images)
    return random_image

# def save_predicted_image(image, predicted_class, save_dir, image_name):
#     """
#     Lưu ảnh vào một thư mục với nhãn dự đoán trong tên file.
    
#     :param image: Ảnh cần lưu.
#     :param predicted_class: Nhãn dự đoán của ảnh.
#     :param save_dir: Thư mục lưu ảnh đã dự đoán.
#     :param image_name: Tên gốc của ảnh.
#     """
#     os.makedirs(save_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
#     save_path = os.path.join(save_dir, f"{image_name}_pred_{predicted_class}.png")
    
#     # Lưu ảnh
#     image.save(save_path)
#     print(f"Đã lưu ảnh dự đoán: {save_path}")

from PIL import ImageDraw, ImageFont


# Dictionary để map các class
class_mapping = {
    0: "class_1",
    1: "class_2",
    2: "class_3",
    3: "class_4"
}

def save_predicted_image(image, predicted_class, save_dir, image_name):
    """
    Lưu ảnh vào một thư mục với nhãn dự đoán trong tên file và thêm tiêu đề lên ảnh.
    
    :param image: Ảnh cần lưu.
    :param predicted_class: Nhãn dự đoán của ảnh.
    :param save_dir: Thư mục lưu ảnh đã dự đoán.
    :param image_name: Tên gốc của ảnh.
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)
    
    # Thêm tiêu đề lên ảnh
    draw = ImageDraw.Draw(image)
    try:
        # Cố gắng tải font nếu có sẵn, nếu không thì dùng font mặc định
        font = ImageFont.truetype("arial.ttf", size=1000)  # Bạn có thể thay đổi đường dẫn và kích thước font
    except IOError:
        font = ImageFont.load_default()
    
    # Lấy tên class từ class_mapping
    class_name = class_mapping.get(predicted_class, "Unknown")
    text = f"Prediction: {class_name}"
    
    # Sử dụng textbbox để tính kích thước của văn bản
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Tính toán vị trí để hiển thị văn bản sao cho ở giữa phần trên của ảnh
    x = (image.width - text_width) / 2
    y = 10  # Khoảng cách từ trên xuống
    
    # Thêm văn bản vào ảnh
    draw.text((x, y), text, font=font, fill="white")
    
    # Đường dẫn lưu ảnh
    save_path = os.path.join(save_dir, f"{image_name}_pred_{class_name}.png")
    
    # Lưu ảnh
    image.save(save_path)
    print(f"Đã lưu ảnh dự đoán: {save_path}")


def show_image(image, predicted_class):
    """
    Hiển thị ảnh cùng với nhãn dự đoán.
    
    :param image: Ảnh cần hiển thị.
    :param predicted_class: Nhãn dự đoán của ảnh.
    """
    plt.imshow(image)
    plt.title(f'Dự đoán: {predicted_class}')
    plt.axis('off')  # Tắt hiển thị trục
    plt.show()

if __name__ == "__main__":
    # Thiết bị (CPU hoặc GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Đường dẫn tới mô hình đã huấn luyện và thư mục chứa ảnh test
    model_path = r'/home/phmlog2103/VScode/Drone_detection/Project_Best_Position/Positions_classification/model_weights/model/best_drone_model_2.pth'
    test_dir = r'/home/phmlog2103/VScode/Drone_detection/Project_Best_Position/Positions_classification/Dataset_Split_class/test'  # Thay bằng đường dẫn tới thư mục test của bạn
    save_dir = r'/home/phmlog2103/VScode/Drone_detection/Project_Best_Position/Positions_classification/predict_data'  # Thư mục để lưu ảnh đã dự đoán

    # Load mô hình
    model = load_model(model_path, num_classes=4, device=device)
    
    # Chọn ngẫu nhiên một ảnh từ tập test
    random_image_path = select_random_image(test_dir)
    image_name = os.path.basename(random_image_path).split('.')[0]
    print(f"Đã chọn ngẫu nhiên ảnh: {random_image_path}")
    
    # Dự đoán cho ảnh
    predicted_class, image = predict_image(random_image_path, model, device)
    print(f"Dự đoán nhãn cho ảnh là: {predicted_class}")

    # Hiển thị ảnh cùng nhãn dự đoán
    show_image(image, predicted_class)

    # Lưu ảnh vào thư mục với nhãn dự đoán
    save_predicted_image(image, predicted_class, save_dir, image_name)
