# import torch
# from train import train_network, save_model, load_model, evaluate_network
# from predict import predict_image_rgb
# from pathlib import Path
# from typing import Tuple

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f'Using device: {device}')

#     image_size_rgb: Tuple[int, int, int] = (3, 100, 100)
#     image_size_depth: Tuple[int, int, int] = (1, 100, 100)

#     # Đường dẫn đến thư mục example_dataset
#     example_dataset_path = Path(r'C:\Users\Hi Windows 11 Home\Documents\Drone_detection\pytorch_image_regession\utils3\Dataset_Split_4')

#     # # Uncomment to train the model
#     # model = train_network(device, 100, image_size_rgb=image_size_rgb, image_size_depth=image_size_depth)
#     # save_model(model, filename='best_modelreg_3.pth')

#     # # Load the trained model
#     # model = load_model(image_size_rgb=image_size_rgb, image_size_depth=image_size_depth, filename='best_modelreg_2.pth')
#     # model.to(device)

#     # # Predict on a new RGB image only
#     # test_rgb_image_path = example_dataset_path / 'test' / 'images' / '000031.png'
    
#     # prediction = predict_image_rgb(model, device, str(test_rgb_image_path), image_size_rgb=image_size_rgb)
#     # print(f'Prediction for {test_rgb_image_path}: {prediction}')

#     # Evaluate the model (still using both RGB and Depth for evaluation)
#     evaluate_network(model, device, image_size_rgb=image_size_rgb, image_size_depth=image_size_depth)




import torch
from train import train_network, save_model, load_model, evaluate_network
from predict import predict_image
from pathlib import Path
from typing import Tuple

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    image_size_rgb: Tuple[int, int, int] = (3, 100, 100)
    image_size_depth: Tuple[int, int, int] = (1, 100, 100)

    # Đường dẫn đến thư mục example_dataset
    example_dataset_path = Path(
        r'/home/phmlog2103/VScode/Drone_detection/Project_Best_Position/Positions_regression/Dataset_Merged')

    # Uncomment to train the model
    # model = train_network(device, 20, image_size_rgb=image_size_rgb, image_size_depth=image_size_depth)
    # save_model(model, filename='best_modelreg_5.pth')

    # Load the trained model
    model = load_model(image_size_rgb=image_size_rgb, image_size_depth=image_size_depth, filename='best_modelreg_8.pth')
    model.to(device)
    
    # Predict on a new pair of RGB and Depth images
    test_rgb_image_path = example_dataset_path / 'test' / 'images' / '000003.png'
    test_depth_image_path = example_dataset_path / 'test' / 'depths' / '000003.png'
    
    prediction = predict_image(model, device, str(test_rgb_image_path), str(test_depth_image_path),
                               image_size_rgb=image_size_rgb, image_size_depth=image_size_depth)
    print(f'Prediction for {test_rgb_image_path}: {prediction}')

    # Evaluate the model
    evaluate_network(model, device, image_size_rgb=image_size_rgb, image_size_depth=image_size_depth)
