import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

def ensure_directories_exist(dirs):
    """
    Đảm bảo rằng các thư mục đích tồn tại, nếu chưa có thì tạo mới.
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def split_data(source_dir, train_dir, val_dir, test_dir, csv_file, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Chia dữ liệu từ thư mục nguồn thành các thư mục train, val, test dựa trên tỷ lệ và
    giữ nguyên cấu trúc thư mục class_1, class_2, class_3, class_4.
    
    Đồng thời giữ lại file CSV tổng trong thư mục gốc bên ngoài (output_dir).
    
    source_dir: Thư mục chứa ảnh gốc theo class (ví dụ class_1, class_2, ...)
    train_dir: Thư mục đích chứa ảnh huấn luyện
    val_dir: Thư mục đích chứa ảnh xác nhận (validation)
    test_dir: Thư mục đích chứa ảnh kiểm tra (test)
    csv_file: Đường dẫn tới file CSV chứa thông tin tọa độ
    output_dir: Thư mục tổng dataset_split (nơi để file CSV tổng)
    split_ratio: Tuple chứa tỷ lệ train, val, test (ví dụ: (0.7, 0.15, 0.15))
    """
    # Tạo các thư mục đích train, val, test
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]  # Chỉ lấy các thư mục (class_1, class_2, ...)

    # Tạo các thư mục tương ứng cho train, val, test
    for class_name in classes:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        # Đảm bảo các thư mục đích tồn tại
        ensure_directories_exist([train_class_dir, val_class_dir, test_class_dir])
        
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)  # Lấy danh sách ảnh trong từng class

        # Kiểm tra nếu không có ảnh nào trong thư mục thì bỏ qua
        if len(images) == 0:
            print(f"Thư mục {class_name} không có ảnh nào, bỏ qua.")
            continue
        
        # Chia ảnh thành train, val, test
        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - split_ratio[0]), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=42)

        # Sao chép ảnh vào thư mục train
        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

        # Sao chép ảnh vào thư mục val
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))

        # Sao chép ảnh vào thư mục test
        for img in test_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

        print(f"Đã chia {class_name} thành train, val, test.")

    # Sao chép file CSV vào thư mục tổng (bên ngoài train, val, test)
    # shutil.copy(csv_file, output_dir)
    # print(f"Đã sao chép file CSV tổng vào thư mục: {output_dir}")

def main():
    # Đường dẫn tới thư mục tổng chứa ảnh và các class
    source_dir = r'C:\Users\Hi Windows 11 Home\Documents\Drone_detection\pytorch_image_regession\Dataset_combined'  # Thư mục chứa các class_1, class_2, ...
    csv_file = r'C:\Users\Hi Windows 11 Home\Documents\Drone_detection\pytorch_image_regession\Dataset_combined\Cam_positions.csv'  # Đường dẫn tới file CSV chung
    
    # Đường dẫn tới thư mục tổng dataset_split và các thư mục con
    output_dir = r'C:\Users\Hi Windows 11 Home\Documents\Drone_detection\pytorch_image_regession\utils\example_dataset\train.csv\Dataset_Split'  # Thư mục tổng chứa train, val, test
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    # Chia dữ liệu thành train, val, test với tỷ lệ 70%, 15%, 15%
    split_data(source_dir, train_dir, val_dir, test_dir, csv_file, output_dir, split_ratio=(0.7, 0.15, 0.15))

if __name__ == "__main__":
    main()
