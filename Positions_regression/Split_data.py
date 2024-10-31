# import os
# import shutil
# import pandas as pd
# from sklearn.model_selection import train_test_split

# def rename_images_in_csv(df, folder_name):
#     """
#     Rename image_path in the DataFrame to match the desired format (e.g., Dataset_split/train/images/000001.png).
#     """
#     df['image_path'] = df['image_path'].apply(lambda x: f"Dataset_Split_4/{folder_name}/images/{int(x):06d}.png")

# def move_images_and_save_csv(df, folder, csv_name, image_dir, csv_output_dir, folder_name):
#     """
#     Move images to corresponding folders and create new CSVs that match the images in the folder.
#     The image_path column is renamed to include the folder path.
#     """
#     # Rename image paths in CSV to ensure correct format with folder name
#     rename_images_in_csv(df, folder_name)
    
#     # Create images subfolder inside the current folder (e.g., train/images)
#     images_folder = os.path.join(folder, 'images')
#     os.makedirs(images_folder, exist_ok=True)  # Create the images folder if it doesn't exist
    
#     # Sort the DataFrame based on the order of image_path
#     df_sorted = df.sort_values(by='image_path').reset_index(drop=True)
    
#     # Save the CSV file to the output directory (not the folder of images)
#     csv_path = os.path.join(csv_output_dir, f'{csv_name}.csv')
#     df_sorted.to_csv(csv_path, index=False)
    
#     missing_images = []
    
#     # Loop through the sorted DataFrame to move images in the same order as in CSV
#     for idx, row in df_sorted.iterrows():
#         image_name = row['image_path'].split('/')[-1]  # Extract the image name from the path (000001.png)
#         src_image = os.path.join(image_dir, image_name)  # Find the image
        
#         if os.path.exists(src_image):
#             dst_image = os.path.join(images_folder, image_name)  # Move image to train/images/
#             shutil.copy(src_image, dst_image)
#         else:
#             missing_images.append(image_name)
    
#     if missing_images:
#         print(f"Images not found: {missing_images}")

# def split_dataset(csv_file, output_dir, image_dir, csv_output_dir):
#     # Load the CSV file
#     cam_positions_df = pd.read_csv(csv_file)

#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)

#     # Set up paths for train, test, and val folders inside the new directory
#     train_dir = os.path.join(output_dir, 'train')
#     test_dir = os.path.join(output_dir, 'test')
#     val_dir = os.path.join(output_dir, 'val')

#     # Create directories for train, test, and val if they don't exist
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)

#     # Split the dataset into train (70%), test (15%), and val (15%)
#     train_val_df, test_df = train_test_split(cam_positions_df, test_size=0.15, random_state=42)
#     train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)  # 0.1765 * 85% ≈ 15%

#     # Move images and create CSVs for each subset, save CSVs to the parent output directory
#     move_images_and_save_csv(train_df, train_dir, 'train', image_dir, csv_output_dir, 'train')
#     move_images_and_save_csv(test_df, test_dir, 'test', image_dir, csv_output_dir, 'test')
#     move_images_and_save_csv(val_df, val_dir, 'val', image_dir, csv_output_dir, 'val')

# if __name__ == "__main__":
#     # Define paths
#     csv_file = r'C:\Users\Hi Windows 11 Home\Documents\Drone_detection\pytorch_image_regession\Dataset_combined_2\Cam_positions_2.csv'
#     base_dir = r'C:\Users\Hi Windows 11 Home\Documents\Drone_detection\pytorch_image_regession\Dataset_combined_2'
#     image_dir = os.path.join(base_dir, 'Cam_rgb_2')
    
#     # Create a new folder to save the splits
#     output_dir = r'/home/phmlog2103/VScode/Drone_detection/pytorch_image_regession/utils3\Dataset_Split_5'

#     # Define the folder where CSVs will be saved (the parent folder)
#     csv_output_dir = r'/home/phmlog2103/VScode/Drone_detection/pytorch_image_regession/utils3\Dataset_Split_5'

#     # Call the function to split dataset and save to the new directory
#     split_dataset(csv_file, output_dir, image_dir, csv_output_dir)






import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def rename_images_in_csv(df, folder_name):
    """
    Rename image_path_rgb and image_path_depth in the DataFrame to match the desired format.
    """
    # df['image_path_rgb'] = df['image_path_rgb'].apply(lambda x: f"Dataset_Split_medical/{folder_name}/images/{int(x):06d}.jpg")
    # df['image_path_depth'] = df['image_path_depth'].apply(lambda x: f"Dataset_Split_medical/{folder_name}/depths/{int(x):06d}.jpg")

    df['image_path_rgb'] = df['image_path_rgb'].apply(lambda x: f"Dataset_Split_medical/{folder_name}/images/{int(x.split('.')[0]):06d}.png")
    df['image_path_depth'] = df['image_path_depth'].apply(lambda x: f"Dataset_Split_medical/{folder_name}/depths/{int(x.split('.')[0]):06d}.png")


def move_images_and_save_csv(df, folder, csv_name, image_dir_rgb, image_dir_depth, csv_output_dir, folder_name):
    """
    Move images to corresponding folders and create new CSVs that match the images in the folder.
    The image_path_rgb and image_path_depth columns are renamed to include the folder path.
    """
    # Rename image paths in CSV to ensure correct format with folder name
    rename_images_in_csv(df, folder_name)
    
    # Reorder columns to place image_path_depth as the second column
    df = df[['image_path_rgb', 'image_path_depth'] + [col for col in df.columns if col not in ['image_path_rgb', 'image_path_depth']]]
    
    # Create images and depth subfolders inside the current folder (e.g., train/images, train/depth)
    images_folder = os.path.join(folder, 'images')
    depth_folder = os.path.join(folder, 'depths')
    os.makedirs(images_folder, exist_ok=True)  # Create the images folder if it doesn't exist
    os.makedirs(depth_folder, exist_ok=True)  # Create the depth folder if it doesn't exist
    
    # Sort the DataFrame based on the order of image_path_rgb
    df_sorted = df.sort_values(by='image_path_rgb').reset_index(drop=True)
    
    # Save the CSV file to the output directory (not the folder of images)
    csv_path = os.path.join(csv_output_dir, f'{csv_name}.csv')
    df_sorted.to_csv(csv_path, index=False)
    
    missing_images = []
    missing_depth_images = []
    
    # Loop through the sorted DataFrame to move images in the same order as in CSV
    for idx, row in df_sorted.iterrows():
        # Process RGB images
        image_name_rgb = row['image_path_rgb'].split('/')[-1]  # Extract the image name from the path (000001.png)
        src_image_rgb = os.path.join(image_dir_rgb, image_name_rgb)  # Find the RGB image
        
        if os.path.exists(src_image_rgb):
            dst_image_rgb = os.path.join(images_folder, image_name_rgb)  # Move image to train/images/
            shutil.copy(src_image_rgb, dst_image_rgb)
        else:
            missing_images.append(image_name_rgb)
        
        # Process depth images
        image_name_depth = row['image_path_depth'].split('/')[-1]  # Extract the depth image name from the path (000001.png)
        src_image_depth = os.path.join(image_dir_depth, image_name_depth)  # Find the depth image
        
        if os.path.exists(src_image_depth):
            dst_image_depth = os.path.join(depth_folder, image_name_depth)  # Move depth image to train/depth/
            shutil.copy(src_image_depth, dst_image_depth)
        else:
            missing_depth_images.append(image_name_depth)
    
    if missing_images:
        print(f"RGB images not found: {missing_images}")
    if missing_depth_images:
        print(f"Depth images not found: {missing_depth_images}")

def split_dataset(csv_file, output_dir, image_dir_rgb, image_dir_depth, csv_output_dir):
    # Load the CSV file
    cam_positions_df = pd.read_csv(csv_file)

    # Rename columns for RGB and depth image paths
    cam_positions_df.rename(columns={'image_path': 'image_path_rgb'}, inplace=True)
    cam_positions_df['image_path_depth'] = cam_positions_df['image_path_rgb']  # Duplicate column for depth paths

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up paths for train, test, and val folders inside the new directory
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    val_dir = os.path.join(output_dir, 'val')

    # Create directories for train, test, and val if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Split the dataset into train (70%), test (15%), and val (15%)
    train_val_df, test_df = train_test_split(cam_positions_df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)  # 0.1765 * 85% ≈ 15%

    # Move images and create CSVs for each subset, save CSVs to the parent output directory
    move_images_and_save_csv(train_df, train_dir, 'train', image_dir_rgb, image_dir_depth, csv_output_dir, 'train')
    move_images_and_save_csv(test_df, test_dir, 'test', image_dir_rgb, image_dir_depth, csv_output_dir, 'test')
    move_images_and_save_csv(val_df, val_dir, 'val', image_dir_rgb, image_dir_depth, csv_output_dir, 'val')

if __name__ == "__main__":
    # Define paths
    csv_file = r'/home/phmlog2103/VScode/project_recognition/osteoporosis/utils1/Dataset_new/updated_dataset_info.csv'
    base_dir = r'/home/phmlog2103/VScode/project_recognition/osteoporosis/utils1/Dataset_new'
    image_dir_rgb = os.path.join(base_dir, 'rgb')
    image_dir_depth = os.path.join(base_dir, 'depth')
    
    # Create a new folder to save the splits
    output_dir = r'/home/phmlog2103/VScode/project_recognition/osteoporosis/utils1/Dataset_Split_medical'

    # Define the folder where CSVs will be saved (the parent folder)
    csv_output_dir = r'/home/phmlog2103/VScode/project_recognition/osteoporosis/utils1/Dataset_Split_medical'

    # Call the function to split dataset and save to the new directory
    split_dataset(csv_file, output_dir, image_dir_rgb, image_dir_depth, csv_output_dir)
