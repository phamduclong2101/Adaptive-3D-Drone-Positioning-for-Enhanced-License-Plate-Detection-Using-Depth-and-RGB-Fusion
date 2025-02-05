import os
import shutil
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.prediction import visualize_object_predictions
from IPython.display import Image
from numpy import asarray
import cv2

def setup_directories(base_output_folder, temp_output_folder, high_conf_folder):
    os.makedirs(base_output_folder, exist_ok=True)
    os.makedirs(temp_output_folder, exist_ok=True)
    os.makedirs(high_conf_folder, exist_ok=True)

def download_and_load_model(yolov8_model_path, confidence_threshold=0.5):
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=yolov8_model_path,
        confidence_threshold=confidence_threshold,
        device="cuda:0"
    )
    return detection_model

def process_image(input_path, detection_model, temp_output_folder, confidence_threshold):
    # Dự đoán và lưu hình ảnh kết quả vào thư mục tạm
    result = get_sliced_prediction(
        image=input_path,
        detection_model=detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    result.export_visuals(temp_output_folder)
    
    # Kiểm tra nếu có ít nhất một đối tượng có confidence >= ngưỡng
    high_confidence_detected = any(
        float(detection.score.value) >= confidence_threshold for detection in result.object_prediction_list
    )
    return high_confidence_detected

def process_folder_structure(base_input_folder, base_output_folder, temp_output_folder, high_conf_folder, detection_model, confidence_threshold=0.85):
    global_index = 1
    for cam_folder in sorted(os.listdir(base_input_folder)):
        cam_folder_path = os.path.join(base_input_folder, cam_folder)
        
        if os.path.isdir(cam_folder_path) and cam_folder.startswith("Cam_"):
            for rgb_folder in sorted(os.listdir(cam_folder_path)):
                rgb_folder_path = os.path.join(cam_folder_path, rgb_folder)
                
                if os.path.isdir(rgb_folder_path) and rgb_folder.startswith("Cam_rgb_"):
                    output_rgb_folder = os.path.join(base_output_folder, cam_folder, rgb_folder)
                    os.makedirs(output_rgb_folder, exist_ok=True)
                    
                    for filename in sorted(os.listdir(rgb_folder_path)):
                        if filename.endswith(".jpg") or filename.endswith(".png"):
                            input_path = os.path.join(rgb_folder_path, filename)
                            
                            high_confidence_detected = process_image(
                                input_path, detection_model, temp_output_folder, confidence_threshold
                            )

                            temp_file = os.path.join(temp_output_folder, "prediction_visual.png")
                            new_filename = f"{str(global_index).zfill(6)}.jpg"
                            output_path = os.path.join(output_rgb_folder, new_filename)
                            shutil.move(temp_file, output_path)

                            if high_confidence_detected:
                                high_conf_output_path = os.path.join(high_conf_folder, new_filename)
                                shutil.copy(output_path, high_conf_output_path)

                            global_index += 1

def main():
    # Xử lý thư mục và ảnh
    process_folder_structure(
        base_input_folder,
        base_output_folder,
        temp_output_folder,
        high_conf_folder,
        detection_model,
        confidence_threshold
    )

    print("Dự đoán và lưu kết quả hoàn tất!")

if __name__ == "__main__":
    # Cấu hình các đường dẫn và ngưỡng
    yolov8_model_path = r"/home/phmlog2103/VScode/Drone_detection/yolo_val/license_plate_weights.pt"
    base_input_folder = r"/home/phmlog2103/VScode/Drone_detection/yolo_val/Ground_truth"
    base_output_folder = r"/home/phmlog2103/VScode/Drone_detection/yolo_val/Ground_truth_result"
    temp_output_folder = r"/home/phmlog2103/VScode/Drone_detection/yolo_val/temp_result"
    high_conf_folder = r"/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/confidence(0.9)"
    confidence_threshold = 0.85

    # Thiết lập thư mục và tải model
    setup_directories(base_output_folder, temp_output_folder, high_conf_folder)
    detection_model = download_and_load_model(yolov8_model_path, confidence_threshold=0.5)

    # Gọi hàm main để bắt đầu xử lý
    main()
