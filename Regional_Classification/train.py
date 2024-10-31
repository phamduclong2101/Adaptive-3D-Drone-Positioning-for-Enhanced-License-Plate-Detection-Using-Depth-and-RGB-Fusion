# import torch
# import time
# import copy

# def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):
    
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     history = {
#         'train_loss': [],
#         'val_loss': [],
#         'train_acc': [],
#         'val_acc': []
#     }

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch + 1}/{num_epochs}')
#         print('-' * 10)
        
#         epoch_start = time.time()  # Bắt đầu đo thời gian mỗi epoch

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#                         scheduler.step()  # Cập nhật scheduler chỉ trong giai đoạn train

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             if phase == 'train':
#                 history['train_loss'].append(epoch_loss)
#                 history['train_acc'].append(epoch_acc.item())
#             else:
#                 history['val_loss'].append(epoch_loss)
#                 history['val_acc'].append(epoch_acc.item())

#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         epoch_time_elapsed = time.time() - epoch_start
#         print(f'Epoch completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:4f}')

#     model.load_state_dict(best_model_wts)
    
#     return model, history






import torch
import time
import copy

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):
    """
    Huấn luyện mô hình và lưu lại lịch sử về loss và accuracy.
    
    :param model: Mô hình cần huấn luyện.
    :param criterion: Hàm mất mát.
    :param optimizer: Bộ tối ưu.
    :param scheduler: Bộ điều chỉnh learning rate.
    :param dataloaders: DataLoader cho tập train và val.
    :param dataset_sizes: Kích thước của từng dataset.
    :param num_epochs: Số epoch huấn luyện.
    :param device: Thiết bị (CPU/GPU).
    :return: Mô hình tốt nhất và lịch sử huấn luyện.
    """
    since = time.time()

    # Copy mô hình tốt nhất
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Khởi tạo history để lưu quá trình huấn luyện
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        epoch_start = time.time()  # Bắt đầu đo thời gian mỗi epoch

        # Mỗi epoch có cả giai đoạn train và val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Đặt mô hình ở chế độ huấn luyện
            else:
                model.eval()   # Đặt mô hình ở chế độ đánh giá

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua từng batch trong dataloader
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Chỉ theo dõi gradient trong quá trình train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Lưu lịch sử loss và accuracy cho từng epoch
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Lưu lại mô hình tốt nhất
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

        print()

        epoch_time_elapsed = time.time() - epoch_start
        print(f'Epoch completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load lại trọng số của mô hình tốt nhất
    model.load_state_dict(best_model_wts)
    
    # Trả về mô hình và lịch sử huấn luyện
    return model, history