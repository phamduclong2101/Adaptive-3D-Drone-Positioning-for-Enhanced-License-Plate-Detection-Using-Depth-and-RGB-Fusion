
# from ultralytics import YOLO
# import cv2
# import os

# def detect_objects_eval(model_path, base_input_folder, base_output_folder):
    
#     model = YOLO(model_path)

#     for cam_folder in os.listdir(base_input_folder):
#         cam_folder_path = os.path.join(base_input_folder, cam_folder)
        
#         if os.path.isdir(cam_folder_path) and cam_folder.startswith("Cam_"):
            
#             for rgb_folder in os.listdir(cam_folder_path):
#                 rgb_folder_path = os.path.join(cam_folder_path, rgb_folder)
                
#                 if os.path.isdir(rgb_folder_path) and rgb_folder.startswith("Cam_rgb_"):
                    
#                     output_folder = os.path.join(base_output_folder, cam_folder, f'Result_{cam_folder.lower()}')
#                     os.makedirs(output_folder, exist_ok=True)
                    
#                     for filename in os.listdir(rgb_folder_path):
#                         if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter for image files
#                             image_path = os.path.join(rgb_folder_path, filename)
#                             image = cv2.imread(image_path)
                            
#                             results = model(image)
                            
#                             annotated_image = results[0].plot()
#                             output_path = os.path.join(output_folder, filename)
#                             cv2.imwrite(output_path, annotated_image)

#     print("Detection completed and images saved to the respective folders.")

# def save_high_confidence_images(model_path, base_output_folder, high_conf_folder, confidence_threshold=0.9):
    
#     model = YOLO(model_path)

#     for cam_folder in os.listdir(base_output_folder):
#         if cam_folder.startswith("Cam_"):
#             result_folder = os.path.join(base_output_folder, cam_folder, f'Result_{cam_folder.lower()}')
            
#             high_conf_output_folder = os.path.join(high_conf_folder, cam_folder)
#             os.makedirs(high_conf_output_folder, exist_ok=True)
            
#             for filename in os.listdir(result_folder):
#                 image_path = os.path.join(result_folder, filename)
#                 image = cv2.imread(image_path)
                
#                 results = model(image)
                
#                 for result in results:
#                     if any(detection.conf > confidence_threshold for detection in result.boxes):
#                         high_conf_image_path = os.path.join(high_conf_output_folder, filename)
#                         cv2.imwrite(high_conf_image_path, image)
#                         break  

#     print("High confidence images saved to:", high_conf_folder)

# if __name__ == "__main__":

#     model_path = r'/home/phmlog2103/VScode/yolo_val/license_plate_weights.pt'
#     base_input_folder = r'/home/phmlog2103/VScode/yolo_val/Ground_truth'
#     base_output_folder = r'/home/phmlog2103/VScode/yolo_val/Ground_truth_result'
#     high_conf_folder = r'/home/phmlog2103/VScode/yolo_val/Ground_truth_result/HighConfidenceImages'

#     detect_objects_eval(model_path, base_input_folder, base_output_folder)

#     save_high_confidence_images(model_path, base_output_folder, high_conf_folder)


# from ultralytics import YOLO
# import cv2
# import os
# import numpy as np
# from sahi.predict import get_sliced_prediction
# from sahi.models.base import DetectionModel
# from sahi.utils.cv import read_image_as_pil

# class Yolov8Wrapper(DetectionModel):
#     def __init__(self, model_path, confidence_threshold=0.5, device="cpu"):
#         self.model = YOLO(model_path)
#         self.confidence_threshold = confidence_threshold
#         self.device = device
#         self.model.to(self.device)
    
#     def perform_inference(self, image):
#         # Run YOLOv8 model inference
#         results = self.model(image)
#         return results
    
#     def convert_original_predictions(self, results, image_size, shift_amount=(0, 0)):
#         # Convert YOLOv8 results to SAHI format
#         object_predictions = []
#         for box in results[0].boxes:
#             if box.conf > self.confidence_threshold:
#                 # Box coordinates and confidence
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 score = box.conf.item()
#                 category_id = int(box.cls.item())
#                 # Apply shift_amount if slicing was applied (default is (0,0) if no slicing)
#                 x1 += shift_amount[0]
#                 y1 += shift_amount[1]
#                 x2 += shift_amount[0]
#                 y2 += shift_amount[1]
#                 object_predictions.append({
#                     "bbox": [x1, y1, x2 - x1, y2 - y1],
#                     "score": score,
#                     "category_id": category_id,
#                     "category_name": str(category_id)
#                 })
#         return object_predictions

# def detect_objects_eval_with_sahi_full_image(model_path, base_input_folder, base_output_folder, high_conf_folder, confidence_threshold=0.8):
#     # Load YOLOv8 model with custom wrapper
#     detection_model = Yolov8Wrapper(model_path, confidence_threshold=confidence_threshold)

#     for cam_folder in os.listdir(base_input_folder):
#         cam_folder_path = os.path.join(base_input_folder, cam_folder)
        
#         if os.path.isdir(cam_folder_path) and cam_folder.startswith("Cam_"):
            
#             for rgb_folder in os.listdir(cam_folder_path):
#                 rgb_folder_path = os.path.join(cam_folder_path, rgb_folder)
                
#                 if os.path.isdir(rgb_folder_path) and rgb_folder.startswith("Cam_rgb_"):
                    
#                     output_folder = os.path.join(base_output_folder, cam_folder, f'Result_{cam_folder.lower()}')
#                     high_conf_output_folder = os.path.join(high_conf_folder, cam_folder, f'HighConfidence_{cam_folder.lower()}')
                    
#                     # Tạo thư mục nếu chưa tồn tại
#                     os.makedirs(output_folder, exist_ok=True)
#                     os.makedirs(high_conf_output_folder, exist_ok=True)
                    
#                     for filename in os.listdir(rgb_folder_path):
#                         if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter for image files
#                             image = os.path.join(rgb_folder_path, filename)
                            
#                             # Use SAHI to perform prediction on the full image (without slicing)
#                             results = get_sliced_prediction(
#                                 image=image,
#                                 detection_model=detection_model,
#                                 slice_height=10000,  # Set very large slice height and width to cover the full image
#                                 slice_width=10000,
#                                 overlap_height_ratio=0,  # Set overlap to 0 to avoid slicing
#                                 overlap_width_ratio=0
#                             )

#                             high_confidence_detected = False

#                             # Kiểm tra nếu bất kỳ dự đoán nào có độ tự tin lớn hơn ngưỡng
#                             for detection in results.object_prediction_list:
#                                 if detection.score > confidence_threshold:
#                                     high_confidence_detected = True
#                                     break

#                             # Lưu ảnh vào thư mục phù hợp
#                             annotated_image = results.get_visualization()  # Get the annotated image from SAHI
#                             output_path = os.path.join(output_folder, filename)
#                             cv2.imwrite(output_path, annotated_image)

#                             # Nếu có dự đoán với độ tự tin cao, lưu ảnh vào thư mục high_conf
#                             if high_confidence_detected:
#                                 high_conf_output_path = os.path.join(high_conf_output_folder, filename)
#                                 cv2.imwrite(high_conf_output_path, annotated_image)

#     print("Detection completed with SAHI full image prediction and YOLOv8 weights. Images saved to the respective folders.")

# if __name__ == "__main__":

#     # Path to your pre-trained YOLOv8 weights
#     model_path = r'/home/phmlog2103/VScode/Drone_detection/yolo_val/license_plate_weights.pt'
#     base_input_folder = r'/home/phmlog2103/VScode/Drone_detection/yolo_val/Ground_truth'
#     base_output_folder = r'/home/phmlog2103/VScode/Drone_detection/yolo_val/Ground_truth_result'
#     high_conf_folder = r'/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages/confidence(0.85)'

#     # Perform object detection using SAHI full image prediction and pre-trained YOLOv8 model
#     detect_objects_eval_with_sahi_full_image(model_path, base_input_folder, base_output_folder, high_conf_folder)

# import os
# from PIL import Image

# def rename_image(input_dir, output_dir):

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     file_list = sorted([f for f in os.listdir(input_dir)],
#                        key=lambda x : x.split(".")[0])

#     for file_name in file_list:
#         img_path = os.path.join(input_dir, file_name)
#         img = Image.open(img_path).convert("RGB")

#         original_image = os.path.splitext(file_name)[0]
#         new_name = f"00{original_image}.jpg"

#         img.save(os.path.join(output_dir, new_name), "JPEG")
#         print(f"Converted {file_name} to {new_name}")


# if __name__ == "__main__":

#     input_dir = r"/home/phmlog2103/VScode/Drone_detection/yolo_val/Ground_truth/Cam_rgb_000005"
#     output_dir = r"/home/phmlog2103/VScode/Drone_detection/yolo_val/Ground_truth/Cam_rgb_000005_new"
#     rename_image(input_dir, output_dir)



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
