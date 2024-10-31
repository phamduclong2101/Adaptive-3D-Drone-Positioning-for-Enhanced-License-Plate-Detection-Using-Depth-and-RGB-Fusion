import os
import pandas as pd
import numpy as np



class DatasetProcessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
    
    def rotate_point(self, camera_x, camera_y, object_x, object_y, object_rotation_z):

        dx = camera_x - object_x
        dy = camera_y - object_y

        theta_adjusted = -object_rotation_z + np.radians(180)

        rotation_matrix = np.array([[np.cos(theta_adjusted), -np.sin(theta_adjusted)],  
                                    [np.sin(theta_adjusted), np.cos(theta_adjusted)]])

        rotated_coords = np.dot(rotation_matrix, np.array([dx, dy]))
        return rotated_coords[0], rotated_coords[1]

    def classify_camera_position(self, camera_x, camera_y, camera_z, object_x, object_y, object_z, object_rotation_z):
        x_local, y_local = self.rotate_point(camera_x, camera_y, object_x, object_y, object_rotation_z)
        
        if x_local > 0 and y_local > 0:
            if x_local < y_local:
                return "Class_2"  
            else:
                return "Class_2"  
        elif x_local < 0 and y_local > 0:
            if -x_local < y_local:
                return "Class_1"  
            else:
                return "Class_1"  
        elif x_local > 0 and y_local < 0:
            if x_local < -y_local:
                return "Class_3"  
            else:
                return "Class_3"  
        elif x_local < 0 and y_local < 0:
            if -x_local < -y_local:
                return "Class_4"  
            else:
                return "Class_4"  
        else:
            return "undefined"

    def process_csv(self, file_path):

        df = pd.read_csv(file_path, dtype={'frame_index': str})

        required_columns = ['Camera_X', 'Camera_Y', 'Camera_Z', 'Object_X', 'Object_Y', 'Object_Z', 'Object_Rotation_Z', 'frame_index']
        if not all(column in df.columns for column in required_columns):
            print(f"Thiếu cột cần thiết trong file: {file_path}")
            return

        df['label'] = df.apply(lambda row: self.classify_camera_position(
            row['Camera_X'], row['Camera_Y'], row['Camera_Z'],
            row['Object_X'], row['Object_Y'], row['Object_Z'],
            np.radians(row['Object_Rotation_Z'])
        ), axis=1)

        df['frame_index'] = df['frame_index'].apply(lambda x: f"{int(x):06d}")

        df.to_csv(file_path, index=False)
        print(f"Đã xử lý xong file: {file_path}")

    def process_all_csv_in_dataset(self):

        for i in range(1, 6):
            folder_name = f"Cam_{i:06d}"
            folder_path = os.path.join(self.dataset_dir, folder_name)
            
            if os.path.isdir(folder_path):
                file_name = f"Cam{i:06d}_positions.csv"
                file_path = os.path.join(folder_path, file_name)
                
                if os.path.isfile(file_path):
                    self.process_csv(file_path)
                else:
                    print(f"File không tồn tại: {file_path}")

if __name__ == "__main__":
    dataset_dir = r"/home/phmlog2103/VScode/Drone_detection/yolo_val/HighConfidenceImages"
    
    processor = DatasetProcessor(dataset_dir)
    processor.process_all_csv_in_dataset()

