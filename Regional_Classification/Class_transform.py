# import os
# import pandas as pd
# import shutil

# def ensure_directories_exist(output_dirs):
#     """
#     Đảm bảo rằng các thư mục đích (class_1, class_2, class_3, class_4) tồn tại.
#     Nếu chưa tồn tại, sẽ tạo mới.
#     """
#     for class_name in output_dirs.values():
#         os.makedirs(class_name, exist_ok=True)

# def move_images_to_class_folders(image_dir, csv_file, output_dirs):
#     """
#     Đọc thông tin từ file CSV và di chuyển ảnh từ thư mục tổng đến các thư mục class dựa trên nhãn.
    
#     image_dir: Đường dẫn tới thư mục tổng chứa ảnh
#     csv_file: Đường dẫn tới file CSV chứa tên ảnh và nhãn
#     output_dirs: Dictionary chứa đường dẫn tới các thư mục class
#     """
#     # Đọc file CSV
#     df = pd.read_csv(csv_file)

#     # Lặp qua từng hàng trong file CSV và di chuyển ảnh tới thư mục tương ứng
#     for index, row in df.iterrows():
#         # Chuyển đổi image_name thành chuỗi 6 ký tự và nối thêm đuôi .png
#         img_name = f'{int(row["image_name"]):06}.png'
#         label = row['label']  # Nhãn class (class1, class2, class3, class4)
        
#         # Đường dẫn tới file ảnh gốc trong thư mục tổng
#         img_path = os.path.join(image_dir, img_name)

#         # Kiểm tra xem ảnh có tồn tại hay không
#         if not os.path.exists(img_path):
#             print(f"Ảnh {img_name} không tồn tại")
#             continue  # Bỏ qua ảnh này nếu không tìm thấy file

#         # Đường dẫn tới thư mục đích dựa trên nhãn
#         destination_dir = output_dirs.get(label)  # Lấy thư mục đích từ nhãn

#         if destination_dir:  # Nếu nhãn hợp lệ
#             destination_path = os.path.join(destination_dir, os.path.basename(img_path))
#             shutil.copy(img_path, destination_path)  # Di chuyển ảnh tới thư mục đích với tên giữ nguyên
#             print(f"Coped {img_name} to {destination_dir}")
#         else:
#             print(f"Label {label} không hợp lệ cho ảnh {img_name}")

# def main():
#     # Đường dẫn tới thư mục tổng chứa ảnh RGB và file CSV
#     image_dir = r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/confidence(0.9)/Cam_000001/Cam_rgb'  # Thư mục chứa ảnh tổng
#     csv_file = r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/confidence(0.9)/Cam_000001/Cam_positions.csv'  # Đường dẫn tới file CSV

#     # Đường dẫn tới các thư mục class1, class2, class3, class4
#     output_dirs = {
#         'Class_1': r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/confidence(0.9)/Class_confidence/Class_1',
#         'Class_2': r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/confidence(0.9)/Class_confidence/Class_2',
#         'Class_3': r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/confidence(0.9)/Class_confidence/Class_3',
#         'Class_4': r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/confidence(0.9)/Class_confidence/Class_4',
#     }

#     # Đảm bảo các thư mục class tồn tại
#     ensure_directories_exist(output_dirs)

#     # Di chuyển ảnh từ thư mục tổng tới các thư mục class tương ứng
#     move_images_to_class_folders(image_dir, csv_file, output_dirs)

# if __name__ == "__main__":
#     main()



import os
import pandas as pd
import shutil

def ensure_directories_exist(output_dirs):
    """
    Đảm bảo rằng các thư mục đích (class_1, class_2, class_3, class_4) tồn tại.
    Nếu chưa tồn tại, sẽ tạo mới.
    """
    for class_name in output_dirs.values():
        os.makedirs(class_name, exist_ok=True)

def move_images_to_class_folders(image_dir, csv_file, output_dirs):
    """
    Đọc thông tin từ file CSV và di chuyển ảnh từ thư mục tổng đến các thư mục class dựa trên nhãn.
    
    image_dir: Đường dẫn tới thư mục tổng chứa ảnh
    csv_file: Đường dẫn tới file CSV chứa tên ảnh và nhãn
    output_dirs: Dictionary chứa đường dẫn tới các thư mục class
    """
    # Đọc file CSV
    df = pd.read_csv(csv_file)

    # Đảm bảo cột 'image_name' có định dạng 6 chữ số
    df['image_name'] = df['image_name'].apply(lambda x: str(x).zfill(6))

    # Lặp qua từng hàng trong file CSV và di chuyển ảnh tới thư mục tương ứng
    for index, row in df.iterrows():
        # Lấy tên ảnh đã được định dạng 6 ký tự và nối thêm đuôi .png
        img_name = f'{row["image_name"]}.jpg'
        label = row['label']  # Nhãn class (class1, class2, class3, class4)
        
        # Đường dẫn tới file ảnh gốc trong thư mục tổng
        img_path = os.path.join(image_dir, img_name)

        # Kiểm tra xem ảnh có tồn tại hay không
        if not os.path.exists(img_path):
            print(f"Ảnh {img_name} không tồn tại")
            continue  # Bỏ qua ảnh này nếu không tìm thấy file

        # Đường dẫn tới thư mục đích dựa trên nhãn
        destination_dir = output_dirs.get(label)  # Lấy thư mục đích từ nhãn

        if destination_dir:  # Nếu nhãn hợp lệ
            destination_path = os.path.join(destination_dir, os.path.basename(img_path))
            shutil.copy(img_path, destination_path)  # Di chuyển ảnh tới thư mục đích với tên giữ nguyên
            print(f"Copied {img_name} to {destination_dir}")
        else:
            print(f"Label {label} không hợp lệ cho ảnh {img_name}")

def main():
    # Đường dẫn tới thư mục tổng chứa ảnh RGB và file CSV
    image_dir = r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/Cam_000001/Cam_rgb'  # Thư mục chứa ảnh tổng
    csv_file = r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/Cam_000001/Cam_positions.csv'  # Đường dẫn tới file CSV

    # Đường dẫn tới các thư mục class1, class2, class3, class4
    output_dirs = {
        'Class_1': r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/Cam_000001/Class_confidence/Class_1',
        'Class_2': r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/Cam_000001/Class_confidence/Class_2',
        'Class_3': r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/Cam_000001/Class_confidence/Class_3',
        'Class_4': r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/Cam_000001/Class_confidence/Class_4',
    }

    # Đảm bảo các thư mục class tồn tại
    ensure_directories_exist(output_dirs)

    # Di chuyển ảnh từ thư mục tổng tới các thư mục class tương ứng
    move_images_to_class_folders(image_dir, csv_file, output_dirs)

if __name__ == "__main__":
    main()
