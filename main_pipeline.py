# import torch
# import numpy as np
# from pathlib import Path
# import sys

# # Thiết lập đường dẫn gốc của project
# project_root = Path(r'/home/phmlog2103/VScode/Drone_detection/Project_Best_Position/BPDrone')

# # Thêm các module vào sys.path để có thể import từ các thư mục khác nhau
# sys.path.append(str(project_root / 'Positions_classification'))
# sys.path.append(str(project_root / 'Positions_regression'))
# sys.path.append(str(project_root / 'Ground_truth'))

# # Import từ file prediction.py (nơi chứa load_model)
# from prediction import load_model as load_model_prediction, predict_image, select_random_image, show_image
# # Thêm đường dẫn thư mục Positions_classification
# sys.path.append(str(Path(__file__).resolve().parent / 'Positions_classification'))
# # Sau đó import lại
# from train_classification import train_model
# # Import từ file regression cho Giai đoạn 2
# from regression_main import load_model as load_model_regression, predict_image as predict_image_regression

# from main_based import find_optimal_ground_truth


# def combined_pipeline():
#     # Thiết lập thiết bị
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f'Sử dụng thiết bị: {device}')

#     # Cấu hình các đường dẫn dựa trên đường dẫn gốc
#     images_dir = project_root / 'Dataset' / 'images'
#     depths_dir = project_root / 'Dataset' / 'depths'
#     model_path_prediction = project_root / 'model_weights' / 'best_drone_model_2.pth'
#     model_path_regression = project_root / 'model_weights' / 'best_modelreg_8.pth'
#     ground_truth_dir = project_root / 'Ground_truth'

#     # Giai đoạn 1: Dự đoán class của ảnh RGB
#     model_prediction = load_model_prediction(str(model_path_prediction), num_classes=4, device=device)
    
#     # Chọn ngẫu nhiên một ảnh RGB từ thư mục ảnh
#     rgb_image_path = select_random_image(str(images_dir))
#     predicted_class, image = predict_image(rgb_image_path, model_prediction, device)
#     print(f'Dự đoán class: {predicted_class}')
#     show_image(image, predicted_class)
    
#     # Lưu thông tin class đã dự đoán lại để sử dụng ở giai đoạn 3
#     saved_class = predicted_class
    
#     # Giai đoạn 2: Dự đoán vị trí camera với ảnh RGB và ảnh depth
#     model_regression = load_model_regression(image_size_rgb=(3, 100, 100), image_size_depth=(1, 100, 100), filename=str(model_path_regression))
#     model_regression.to(device)
    
#     # Lấy tên file từ ảnh RGB đã chọn để tìm ảnh depth tương ứng
#     image_name = Path(rgb_image_path).stem  # Lấy tên file (không có phần mở rộng)
#     depth_image_path = depths_dir / f'{image_name}.png'  # Giả sử các ảnh depth có phần mở rộng là .png

#     # Kiểm tra xem file ảnh depth có tồn tại không
#     if not depth_image_path.exists():
#         print(f"Không tìm thấy ảnh depth tương ứng: {depth_image_path}")
#         return

#     # Dự đoán vị trí camera sử dụng ảnh RGB và ảnh depth
#     prediction_position = predict_image_regression(
#         model_regression, device, str(rgb_image_path), str(depth_image_path),
#         image_size_rgb=(3, 100, 100), image_size_depth=(1, 100, 100)
#     )
#     print(f'Dự đoán vị trí cho {rgb_image_path}: {prediction_position}')
    
#     # Lưu vị trí camera đã dự đoán để sử dụng ở giai đoạn 3
#     saved_position = np.array(prediction_position)
    
#     # Giai đoạn 3: Tìm vị trí tối ưu dựa trên class và vị trí dự đoán
#     find_optimal_ground_truth(class_number=saved_class, current_position=saved_position, base_dir=str(ground_truth_dir))

# if __name__ == "__main__":
#     combined_pipeline()

import torch
import numpy as np
from pathlib import Path
import sys
import cv2

# Thiết lập đường dẫn gốc của project
project_root = Path(r'/home/phmlog2103/VScode/Drone_detection/Project_Best_Position/BPDrone')

# Thêm các module vào sys.path để có thể import từ các thư mục khác nhau
sys.path.append(str(project_root / 'Positions_classification'))
sys.path.append(str(project_root / 'Positions_regression'))
sys.path.append(str(project_root / 'Ground_truth'))

# Import từ file prediction.py (nơi chứa load_model)
from prediction import load_model as load_model_prediction, predict_image, select_random_image, show_image
from regression_main import load_model as load_model_regression, predict_image as predict_image_regression
from main_based import find_optimal_ground_truth

# Dictionary để map các class
class_mapping = {
    0: "Class_1",
    1: "Class_2",
    2: "Class_3",
    3: "Class_4"
}

def save_image(image, image_name, output_dir):
    """
    Lưu ảnh đã được dự đoán vào thư mục output.
    
    :param image: Ảnh cần lưu.
    :param image_name: Tên gốc của ảnh.
    :param output_dir: Thư mục để lưu ảnh.
    """
    output_image_path = output_dir / f'{image_name}.png'
    
    # Lưu ảnh ra thư mục output
    cv2.imwrite(str(output_image_path), image)
    print(f"Đã lưu ảnh dự đoán tại: {output_image_path}")

def combined_pipeline(rgb_width=1000):
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use device: {device}')

    # Cấu hình các đường dẫn dựa trên đường dẫn gốc
    images_dir = project_root / 'Dataset' / 'images'
    depths_dir = project_root / 'Dataset' / 'depths'
    model_path_prediction = project_root / 'model_weights' / 'best_drone_model_2.pth'
    model_path_regression = project_root / 'model_weights' / 'best_modelreg_8.pth'
    ground_truth_dir = project_root / 'Ground_truth'
    
    # Thư mục để lưu ảnh đã dự đoán
    output_dir = project_root / 'output_results'  
    output_dir.mkdir(parents=True, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    # Giai đoạn 1: Dự đoán class của ảnh RGB
    model_prediction = load_model_prediction(str(model_path_prediction), num_classes=4, device=device)
    
    # Chọn ngẫu nhiên một ảnh RGB từ thư mục ảnh
    rgb_image_path = select_random_image(str(images_dir))
    image_name = Path(rgb_image_path).stem  # Lấy tên file (không có phần mở rộng)
    predicted_class, image = predict_image(rgb_image_path, model_prediction, device)
    
    # Chuyển đổi chỉ số class thành tên class bằng cách sử dụng class_mapping
    class_name = class_mapping.get(predicted_class, "Unknown Class")
    
    print(f'Dự đoán class: {class_name}')
    
    # Lưu thông tin class đã dự đoán lại để sử dụng ở giai đoạn 3
    saved_class = class_name  # Lưu tên class đã mapping
    
    # Giai đoạn 2: Dự đoán vị trí camera với ảnh RGB và ảnh depth
    model_regression = load_model_regression(image_size_rgb=(3, 100, 100), image_size_depth=(1, 100, 100), filename=str(model_path_regression))
    model_regression.to(device)
    
    # Lấy tên file từ ảnh RGB đã chọn để tìm ảnh depth tương ứng
    depth_image_path = depths_dir / f'{image_name}.png'  # Giả sử các ảnh depth có phần mở rộng là .png

    # Kiểm tra xem file ảnh depth có tồn tại không
    if not depth_image_path.exists():
        print(f"Không tìm thấy ảnh depth tương ứng: {depth_image_path}")
        return

    # Dự đoán vị trí camera sử dụng ảnh RGB và ảnh depth
    prediction_position = predict_image_regression(
        model_regression, device, str(rgb_image_path), str(depth_image_path),
        image_size_rgb=(3, 100, 100), image_size_depth=(1, 100, 100)
    )
    print(f'Dự đoán vị trí cho {rgb_image_path}: {prediction_position}')
    
    # Lưu vị trí camera đã dự đoán để sử dụng ở giai đoạn 3
    saved_position = np.array(prediction_position)
    
    # Giai đoạn 3: Tìm vị trí tối ưu dựa trên class và vị trí dự đoán, đồng thời chèn ảnh RGB đã load vào ảnh ground truth
    combined_image, _ = find_optimal_ground_truth(
        class_number=saved_class, 
        current_position=saved_position, 
        base_dir=str(ground_truth_dir), 
        rgb_image_input_path=rgb_image_path,  # Truyền ảnh RGB đã load từ Giai đoạn 1
        rgb_width=rgb_width  # Điều chỉnh kích cỡ ảnh RGB
    )

    # Kiểm tra xem có tìm được ảnh ground truth không
    if combined_image is None:
        print("Không tìm thấy hoặc không thể đọc được ảnh ground truth.")
    else:
        # Lưu ảnh kết hợp (ảnh ground truth + ảnh RGB input)
        save_image(combined_image, image_name, output_dir)

if __name__ == "__main__":
    combined_pipeline(rgb_width=700)  # Có thể thay đổi kích cỡ ảnh RGB ở đây

