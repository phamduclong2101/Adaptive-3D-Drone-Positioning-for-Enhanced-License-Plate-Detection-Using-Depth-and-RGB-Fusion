import pandas as pd
import numpy as np

class DatasetProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    # Hàm quay tọa độ camera sang hệ tọa độ cục bộ của ô tô (3D)
    def rotate_point(self, object_x, object_y, object_z, camera_x, camera_y, camera_z, Object_rotation_z):
        # Tính khoảng cách tương đối giữa Object và Camera
        dx = camera_x - object_x
        dy = camera_y - object_y
        dz = camera_z - object_z

        # Điều chỉnh góc quay quanh trục Z theo camera
        # theta_adjusted = -Object_rotation_z + np.radians(180)

        theta_adjusted = -Object_rotation_z


        # Ma trận quay trong không gian 3D quanh trục Z
        rotation_matrix = np.array([[np.cos(theta_adjusted), -np.sin(theta_adjusted), 0],
                                    [np.sin(theta_adjusted),  np.cos(theta_adjusted), 0],
                                    [0, 0, 1]])
        
        

        # Vector chứa tọa độ trước khi quay
        relative_position = np.array([dx, dy, dz])

        # Tọa độ mới sau khi quay
        rotated_coords = np.dot(rotation_matrix, relative_position)
        return rotated_coords[0], rotated_coords[1], rotated_coords[2]

    # Hàm để xử lý file CSV
    def process_csv(self):
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(self.file_path)

        # Kiểm tra xem file có chứa các cột cần thiết hay không
        required_columns = ['Camera_X', 'Camera_Y', 'Camera_Z', 'Object_X', 'Object_Y', 'Object_Z', 'Object_Rotation_Z']
        if not all(column in df.columns for column in required_columns):
            print(f"Thiếu cột cần thiết trong file: {self.file_path}")
            return

        # Thực hiện tính toán cho từng dòng, chuyển đổi tọa độ object về hệ tọa độ cục bộ của camera
        df[['Cam_X_Local', 'Cam_Y_Local', 'Cam_Z_Local']] = df.apply(
            lambda row: self.rotate_point(
                row['Object_X'], row['Object_Y'], row['Object_Z'],
                row['Camera_X'], row['Camera_Y'], row['Camera_Z'],
                np.radians(row['Object_Rotation_Z'])
            ), axis=1, result_type='expand')
        
        df['Frame_Index'] = df['Frame_Index'].apply(lambda x: f"{int(x):06d}")

        # Lưu lại file CSV với cột tọa độ cục bộ mới
        df.to_csv(self.file_path, index=False)
        print(f"Đã xử lý xong file: {self.file_path}")

if __name__ == "__main__":
    # Nhập đường dẫn tới file CSV
    file_path = r"/home/phmlog2103/VScode/Drone_detection/Blender_Environment/Dataset_combined_toan/Cam_positions_toan.csv"
    
    # Tạo đối tượng DatasetProcessor và xử lý file CSV
    processor = DatasetProcessor(file_path)
    processor.process_csv()


# import pandas as pd
# import numpy as np

# class DatasetProcessor:
#     def __init__(self, file_path):
#         self.file_path = file_path

#     # Hàm quay tọa độ ô tô quanh 3 trục (X, Y, Z)
#     def rotate_point(self, object_x, object_y, object_z, camera_x, camera_y, camera_z, 
#                      camera_rotation_x, camera_rotation_y, camera_rotation_z):
#         # Tính khoảng cách tương đối giữa Object và Camera
#         dx = object_x - camera_x
#         dy = object_y - camera_y
#         dz = object_z - camera_z

#         # Chuyển đổi các góc quay từ độ sang radian
#         theta_x = np.radians(camera_rotation_x)
#         theta_y = np.radians(camera_rotation_y)
#         theta_z = np.radians(camera_rotation_z)

#         # Ma trận quay trong không gian 3D quanh trục X
#         rotation_matrix_x = np.array([[1, 0, 0],
#                                       [0, np.cos(theta_x), -np.sin(theta_x)],
#                                       [0, np.sin(theta_x),  np.cos(theta_x)]])
        
#         # Ma trận quay trong không gian 3D quanh trục Y
#         rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
#                                       [0, 1, 0],
#                                       [-np.sin(theta_y), 0, np.cos(theta_y)]])
        
#         # Ma trận quay trong không gian 3D quanh trục Z
#         rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
#                                       [np.sin(theta_z),  np.cos(theta_z), 0],
#                                       [0, 0, 1]])
        
#         # Kết hợp các ma trận quay: X -> Y -> Z
#         rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

#         # Vector chứa tọa độ trước khi quay
#         relative_position = np.array([dx, dy, dz])

#         # Tọa độ mới sau khi quay
#         rotated_coords = np.dot(rotation_matrix, relative_position)
#         return rotated_coords[0], rotated_coords[1], rotated_coords[2]

#     # Hàm để xử lý file CSV
#     def process_csv(self):
#         # Đọc dữ liệu từ file CSV
#         df = pd.read_csv(self.file_path)

#         # Kiểm tra xem file có chứa các cột cần thiết hay không
#         required_columns = ['Camera_X', 'Camera_Y', 'Camera_Z', 'Object_X', 'Object_Y', 'Object_Z', 
#                             'Camera_Rotation_X', 'Camera_Rotation_Y', 'Camera_Rotation_Z']
#         if not all(column in df.columns for column in required_columns):
#             print(f"Thiếu cột cần thiết trong file: {self.file_path}")
#             return

#         # Thực hiện tính toán cho từng dòng, chuyển đổi tọa độ object về hệ tọa độ cục bộ của camera
#         df[['Object_X_Local', 'Object_Y_Local', 'Object_Z_Local']] = df.apply(
#             lambda row: self.rotate_point(
#                 row['Object_X'], row['Object_Y'], row['Object_Z'],
#                 row['Camera_X'], row['Camera_Y'], row['Camera_Z'],
#                 row['Camera_Rotation_X'], row['Camera_Rotation_Y'], row['Camera_Rotation_Z']
#             ), axis=1, result_type='expand')

#         df['Frame_Index'] = df['Frame_Index'].apply(lambda x: f"{int(x):06d}")

#         # Lưu lại file CSV với cột tọa độ cục bộ mới
#         df.to_csv(self.file_path, index=False)
#         print(f"Đã xử lý xong file: {self.file_path}")

# if __name__ == "__main__":
#     # Nhập đường dẫn tới file CSV
#     file_path = r"C:\Users\Hi Windows 11 Home\Documents\Drone_detection\pytorch_image_regession\Data_combine\position2.csv"
    
#     # Tạo đối tượng DatasetProcessor và xử lý file CSV
#     processor = DatasetProcessor(file_path)
#     processor.process_csv()
