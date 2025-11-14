import numpy as np
import cv2
import torch
from medpy import metric
from scipy.ndimage import zoom


# def calculate_metric_percase(y_pred, y_true):
#     """ pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         if gt.sum() > 0:
#             hd = metric.binary.hd(pred, gt)
#         else:
#             hd = 0

#         eps = 0.0001
#         dilation_ratio=0.005
#         c_pred, h_pred, w_pred = pred.shape
#         y_pred, y_true = np.array(pred), np.array(gt)
#         y_pred, y_true = np.round(pred).astype(int), np.round(gt).astype(int)
#         a_unin_b = np.sum(y_pred[y_true == 1])
#         a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
#         # dice
#         #dice_value = (a_unin_b * 2.0 + eps) / a_plus_b
#         # PPV
#         ppv_value = (a_unin_b * 1.0 + eps) / (np.sum(y_pred) + eps)
#         # sensitivity
#         sen_val = (a_unin_b * 1.0 + eps) / (np.sum(y_true) + eps)
#         #print('ppv_value and sen_val has been calculated')
#         iou = a_unin_b / (a_plus_b - a_unin_b) # a_plus_b里边有eps，所以不加了
        
#         boundary_iou_all = 0.0
#         for i in range(c_pred):
#             gt_boundary = mask_to_boundary(y_true[i], dilation_ratio)
#             dt_boundary = mask_to_boundary(y_pred[i], dilation_ratio)
#             intersection = ((gt_boundary * dt_boundary) > 0).sum()
#             union = ((gt_boundary + dt_boundary) > 0).sum()
#             boundary_iou = intersection / (union + eps)
#             boundary_iou_all += boundary_iou
#         boundary_iou = boundary_iou_all / c_pred """
#     eps = 0.0001
#     c_pred, h_pred, w_pred = y_pred.shape
#     y_pred, y_true = np.array(y_pred), np.array(y_true)
#     y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
#     TP = np.sum(y_pred[y_true == 1])

#     a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
#     #dice
#     dice=(TP * 2.0 + eps) / a_plus_b
#     denominator1=np.sum(y_pred) + eps
#     #PPV
#     ppv=(TP*1.0 + eps) / denominator1
#     denominator2 = np.sum(y_true) + eps
#     #Sen
#     sen=(TP*1.0 + eps) / denominator2
    
#     # hd and hd95
#     if y_pred.sum() > 0 and y_true.sum() > 0:
#         hd = metric.binary.hd(y_pred, y_true)
#         hd95 = metric.binary.hd95(y_pred, y_true)
#     else:
#         hd = 0
#         hd95 = 0
#     # iou
#     a_unin_b = np.sum(y_pred[y_true == 1]) + eps
#     a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
#     iou = a_unin_b / ((a_plus_b - a_unin_b) + eps)

#     # biou
#     boundary_iou_all = 0.0
#     dilation_ratio=0.005
#     for i in range(c_pred):
#         gt_boundary = mask_to_boundary(y_true[i], dilation_ratio)
#         dt_boundary = mask_to_boundary(y_pred[i], dilation_ratio)
#         intersection = ((gt_boundary * dt_boundary) > 0).sum()
#         union = ((gt_boundary + dt_boundary) > 0).sum()
#         boundary_iou = intersection / (union + eps)
#         boundary_iou_all += boundary_iou
#     boundary_iou = boundary_iou_all / c_pred 

#     return dice, hd95, ppv, sen, iou, boundary_iou


def calculate_metric_percase(y_pred, y_true):
    eps = 0.0001
    c_pred, h_pred, w_pred = y_pred.shape
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    # print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
    TP = np.sum(y_pred[y_true == 1])

    a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
    #dice
    dice=(TP * 2.0 + eps) / a_plus_b
    denominator1=np.sum(y_pred) + eps
    #PPV
    ppv=(TP*1.0 + eps) / denominator1
    denominator2 = np.sum(y_true) + eps
    #Sen
    sen=(TP*1.0 + eps) / denominator2
    
    # hd and hd95
    if y_pred.sum() > 0 and y_true.sum() > 0:
        hd = metric.binary.hd(y_pred, y_true)
        hd95 = metric.binary.hd95(y_pred, y_true)
    else:
        hd = 0
        hd95 = 0
    # iou
    a_unin_b = np.sum(y_pred[y_true == 1]) + eps
    a_plus_b = np.sum(y_pred) + np.sum(y_true) + eps
    iou = a_unin_b / ((a_plus_b - a_unin_b) + eps)

    # biou
    boundary_iou_all = 0.0
    dilation_ratio=0.005
    for i in range(c_pred):
        gt_boundary = mask_to_boundary(y_true[i], dilation_ratio)
        dt_boundary = mask_to_boundary(y_pred[i], dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        boundary_iou = intersection / (union + eps)
        boundary_iou_all += boundary_iou
    boundary_iou = boundary_iou_all / c_pred 

    # ASD (Average Surface Distance)
    # Convert to binary for boundary comparison
    # Calculate ASD (Average Surface Distance)
    pred_boundaries = (y_pred > 0).astype(int)
    true_boundaries = (y_true > 0).astype(int)
    
    # Get the surface points (boundary points) of prediction and ground truth
    pred_surface_points = np.array(np.where(pred_boundaries == 1))
    true_surface_points = np.array(np.where(true_boundaries == 1))
    
    # Check if there are no surface points in either prediction or ground truth
    if pred_surface_points.size == 0 or true_surface_points.size == 0:
        # If there are no surface points, ASD is undefined or set to 0
        asd = 0
    else:
        # Calculate distance from predicted surface points to true surface points
        distances = []
        for pred_point in pred_surface_points.T:
            dist = np.min(np.linalg.norm(true_surface_points - pred_point[:, None], axis=0))
            distances.append(dist)
        for true_point in true_surface_points.T:
            dist = np.min(np.linalg.norm(pred_surface_points - true_point[:, None], axis=0))
            distances.append(dist)
    
        # Average Surface Distance
        asd = np.mean(distances) if distances else 0

    return dice, hd95, ppv, sen, iou, boundary_iou, hd,asd


##################################################################
# General util function to get the boundary of a binary mask.
# 该函数用于获取二进制 mask 的边界
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2) # 计算图像对角线长度
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
        
    mask = mask.astype(np.uint8)
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    
    # 因为之前向四周填充了0, 故而这里不再需要四周
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode
##################################################################
def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


# def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
#     # 转换为numpy数组并去除多余的维度
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
#     # 输出图像和标签的形状
#     # print(f"Image shape: {image.shape}")
#     # print(f"Label shape: {label.shape}")
    
#     if len(image.shape) == 3:  # 如果图像有三个维度（例如：batch, height, width）
#         prediction = np.zeros((image.shape[0], 256, 256))  # 预测结果的形状应为 [num_slices, 256, 256]
        
#         for ind in range(image.shape[0]):  # ind 现在是图像的切片索引
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
            
#             # 输出slice的形状
#             # print(f"Slice shape before zoom: {slice.shape}")
            
#             # 对slice进行zoom调整大小
#             slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
#             # print(f"Slice shape after zoom: {slice.shape}")
            
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            
#             # 预测
#             net.eval()
#             with torch.no_grad():
#                 out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
                
#                 # unique_out,counts = np.unique(out,return_counts=True)
#                 # print(f"unique_out:{unique_out},counts:{counts}")

#                 # 输出预测结果的形状
#                 # print(f"Pred shape after softmax and argmax: {out.shape}")
                
#                 # 对预测结果进行zoom调整大小
#                 pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 # print(f"Pred shape after zoom: {pred.shape}")
                
#                 prediction[ind] = pred  # 使用ind作为切片索引，赋值时不会发生形状不匹配
#     else:
#         input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            
#             # 输出预测结果的形状
#             # print(f"Pred shape: {out.shape}")
            
#             prediction = out.cpu().detach().numpy()
    
#     # 输出预测的形状
#     # print(f"Prediction shape: {prediction.shape}")
    
#     # 计算metrics
#     metric_list = []
#     for i in range(1, classes):
#         pred_class = prediction == i
#         label_class = label == i
#         # print(f"Class {i} - Pred sum: {pred_class.sum()}, Label sum: {label_class.sum()}")
#         metric_list.append(calculate_metric_percase(pred_class, label_class))
    
#     return metric_list



# 分类的验证，用于血友病#################################################################
def test_single_volume_classfier(image, label, net, classes, patch_size=[256, 256]):
    image = image.squeeze(0).cpu().detach().numpy()
    label = [tensor.to('cuda') for tensor in label]
    
    total_samples = 0
    correct_predictions_1 = 0
    correct_predictions_2 = 0
    correct_predictions_3 = 0
    correct_predictions_4 = 0
    correct_predictions_5 = 0
    abs_difference_predictions_1 = 0
    abs_difference_predictions_2 = 0
    abs_difference_predictions_3 = 0
    abs_difference_predictions_4 = 0
    abs_difference_predictions_5 = 0
    # for ind in range(image.shape[0]):
        # slice = image[ind, :, :]
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(
        slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        outputs = net(input) 
        # print("outputs :", outputs)
        _, predicted_1 = torch.max(outputs[0].data, 1) # 维度上找到张量中的最大值,_是最大值，predicted 则是最大值所在的索引
        _, predicted_2 = torch.max(outputs[1].data, 1) # 维度上找到张量中的最大值,_是最大值，predicted 则是最大值所在的索引
        _, predicted_3 = torch.max(outputs[2].data, 1) # 维度上找到张量中的最大值,_是最大值，predicted 则是最大值所在的索引
        _, predicted_4 = torch.max(outputs[3].data, 1) # 维度上找到张量中的最大值,_是最大值，predicted 则是最大值所在的索引
        _, predicted_5 = torch.max(outputs[4].data, 1) # 维度上找到张量中的最大值,_是最大值，predicted 则是最大值所在的索引
        total_samples += label[0].size(0)
        # predicted_1是最大值索引，是一个数，用了[0]之后就是一个数，没有影响 
        correct_predictions_1 += (predicted_1[0] == label[0]).sum().item()
        correct_predictions_2 += (predicted_2[0] == label[1]).sum().item()
        correct_predictions_3 += (predicted_3[0] == label[2]).sum().item()
        correct_predictions_4 += (predicted_4[0] == label[3]).sum().item()
        correct_predictions_5 += (predicted_5[0] == label[4]).sum().item()

        # 计算风险等级的平均误差
        abs_difference_predictions_1 += abs(predicted_1[0] - label[0]).sum().item()
        abs_difference_predictions_2 += abs(predicted_2[0] - label[1]).sum().item()
        abs_difference_predictions_3 += abs(predicted_3[0] - label[2]).sum().item()
        abs_difference_predictions_4 += abs(predicted_4[0] - label[3]).sum().item()
        abs_difference_predictions_5 += abs(predicted_5[0] - label[4]).sum().item()
        # print("predicted[0] :", predicted[0])
        # print("label[0] :", label[0])
        # print("correct_predictions :", correct_predictions)

    accuracy_1 = correct_predictions_1 / total_samples
    accuracy_2 = correct_predictions_2 / total_samples
    accuracy_3 = correct_predictions_3 / total_samples
    accuracy_4 = correct_predictions_4 / total_samples
    accuracy_5 = correct_predictions_5 / total_samples
    
    return [accuracy_1,  accuracy_2, accuracy_3, accuracy_4, accuracy_5], \
           [abs_difference_predictions_1,  abs_difference_predictions_2, abs_difference_predictions_3, abs_difference_predictions_4, abs_difference_predictions_5]

# 对于Sen的评测，因为是TP/(TP + FN)
def test_single_volume_classfier_Sen(image, label, net, classes, patch_size=[256, 256]):
    image = image.squeeze(0).cpu().detach().numpy()
    label = [tensor.to('cuda') for tensor in label]
    
    total_samples = 0
    correct_predictions_1 = 0
    correct_predictions_2 = 0
    correct_predictions_3 = 0
    correct_predictions_4 = 0
    correct_predictions_5 = 0
    TP_predictions_1 = 0
    TP_predictions_2 = 0
    TP_predictions_3 = 0
    TP_predictions_4 = 0
    TP_predictions_5 = 0
    TP_plus_FN_predictions_1 = 0
    TP_plus_FN_predictions_2 = 0
    TP_plus_FN_predictions_3 = 0
    TP_plus_FN_predictions_4 = 0
    TP_plus_FN_predictions_5 = 0
    # for ind in range(image.shape[0]):
        # slice = image[ind, :, :]
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(
        slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        outputs = net(input) 
        # print("outputs :", outputs)
        _, predicted_1 = torch.max(outputs[0].data, 1) # 维度上找到张量中的最大值,_是最大值，predicted 则是最大值所在的索引
        _, predicted_2 = torch.max(outputs[1].data, 1) 
        _, predicted_3 = torch.max(outputs[2].data, 1) 
        _, predicted_4 = torch.max(outputs[3].data, 1) 
        _, predicted_5 = torch.max(outputs[4].data, 1) 
        total_samples += label[0].size(0)
        # predicted_1是最大值索引，是一个数，用了[0]之后就是一个数，没有影响 
        correct_predictions_1 += (predicted_1[0] == label[0]).sum().item()
        correct_predictions_2 += (predicted_2[0] == label[1]).sum().item()
        correct_predictions_3 += (predicted_3[0] == label[2]).sum().item()
        correct_predictions_4 += (predicted_4[0] == label[3]).sum().item()
        correct_predictions_5 += (predicted_5[0] == label[4]).sum().item()

        # 计算TP
        """ if predicted_1[0] == label[0] and label[0] != 0:
            TP_predictions_1 += 1
        if predicted_2[0] == label[1] and label[1] != 0:
            TP_predictions_2 += 1
        if predicted_3[0] == label[2] and label[2] != 0:
            TP_predictions_3 += 1
        if predicted_4[0] == label[3] and label[3] != 0:
            TP_predictions_4 += 1
        if predicted_5[0] == label[4] and label[4] != 0:
            TP_predictions_5 += 1 """

        # 为了降低难度，现在只要区分有没有病就行
        if predicted_1[0] != 0 and label[0] != 0:
            TP_predictions_1 += 1
        if predicted_2[0] != 0 and label[1] != 0:
            TP_predictions_2 += 1
        if predicted_3[0] != 0 and label[2] != 0:
            TP_predictions_3 += 1
        if predicted_4[0] != 0 and label[3] != 0:
            TP_predictions_4 += 1
        if predicted_5[0] != 0 and label[4] != 0:
            TP_predictions_5 += 1
        
        # TP + FN 的计算, 如果这个位置不是零那么就加一，会在外部的程序中积累
        if label[0] != 0:
            TP_plus_FN_predictions_1 += 1
        if label[1] != 0:
            TP_plus_FN_predictions_2 += 1
        if label[2] != 0:
            TP_plus_FN_predictions_3 += 1
        if label[3] != 0:
            TP_plus_FN_predictions_4 += 1
        if label[4] != 0:
            TP_plus_FN_predictions_5 += 1

    accuracy_1 = correct_predictions_1 / total_samples # 这五行其实可以不写，因为这个total_samples一直是一，它原本是为了体素而设计的
    accuracy_2 = correct_predictions_2 / total_samples
    accuracy_3 = correct_predictions_3 / total_samples
    accuracy_4 = correct_predictions_4 / total_samples
    accuracy_5 = correct_predictions_5 / total_samples
    
    return [accuracy_1,  accuracy_2, accuracy_3, accuracy_4, accuracy_5], \
           [TP_predictions_1, TP_predictions_2, TP_predictions_3, TP_predictions_4, TP_predictions_5], \
           [TP_plus_FN_predictions_1,  TP_plus_FN_predictions_2, TP_plus_FN_predictions_3, TP_plus_FN_predictions_4, TP_plus_FN_predictions_5]

# 分类的验证，用于吸入综合征#################################################################
def test_single_volume_classfier_Xi_Ru(image, label, net, classes, patch_size=[256, 256]):
    image = image.squeeze(0).cpu().detach().numpy()
    label = [tensor.to('cuda') for tensor in label]
    
    total_samples = 0
    correct_predictions_1 = 0
    # for ind in range(image.shape[0]):
        # slice = image[ind, :, :]
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(
        slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    TP = 0
    FN = 0
    FP = 0
    with torch.no_grad():
        outputs = net(input) 
        # print("outputs :", outputs)
        # _, predicted_1 = torch.max(outputs[0].data) # 维度上找到张量中的最大值,_是最大值，predicted 则是最大值所在的索引
        predicted_1 = torch.argmax(outputs, dim=1)
        #print('label[0]', label[0])  # label[0] tensor(1, device='cuda:0')
        total_samples += 1
        # predicted_1是最大值索引，是一个数，用了[0]之后就是一个数，没有影响 
        # print("predicted_1 :", predicted_1)
        # print("label :", label)
        correct_predictions_1 += (predicted_1[0] == label[0]).sum().item()
        # print("predicted_1[0] :", predicted_1[0])
        # print("label[0] :", label[0])
        #正常案例认为为正例，为0正例
        if predicted_1[0] == 0 and label[0] == 0:
            TP += 1
        if predicted_1[0] == 0 and label[0] == 1:
            FP += 1
        if predicted_1[0] == 1 and label[0] == 0:
            FN += 1

        # print("predicted[0] :", predicted[0])
        # print("label[0] :", label[0])
        # print("correct_predictions :", correct_predictions)

    accuracy_1 = correct_predictions_1 / total_samples
    
    return [accuracy_1, TP, FN, FP]

# 分类的验证，用于吸入综合征,加入AUROC指标#################################################################
def test_single_volume_classfier_Xi_Ru_AUROC(image, label, net, classes, patch_size=[256, 256]):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label[0].to('cuda')  # 简化
    slice = image

    # resize
    from scipy.ndimage import zoom
    x, y = slice.shape
    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

    input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()

    with torch.no_grad():
        outputs = net(input)
        probs = torch.softmax(outputs, dim=1)
        prob_disease = probs[:, 1].cpu().numpy()   # 类别1（患病）的概率
        predicted = torch.argmax(outputs, dim=1)

        # Accuracy 计算
        correct = (predicted[0] == label).item()
        accuracy_1 = correct / 1.0

        # 计算TP/FN/FP（保持医学定义：患病=1为正类）
        TP = int(predicted[0] == 1 and label == 1)
        FP = int(predicted[0] == 1 and label == 0)
        FN = int(predicted[0] == 0 and label == 1)

    return [accuracy_1, TP, FN, FP], label.item(), prob_disease[0]


# 分类的验证，用于肺透明膜病#################################################################
def test_single_volume_classfier_Fei_Tou(image, label, net, classes, patch_size=[256, 256]):
    image = image.squeeze(0).cpu().detach().numpy()
    label = [tensor.to('cuda') for tensor in label]
    
    total_samples = 0
    correct_predictions_1 = 0
    # for ind in range(image.shape[0]):
        # slice = image[ind, :, :]
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(
        slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    TP = 0
    FN = 0
    FP = 0
    with torch.no_grad():
        # features,outputs = net(input) 
        outputs = net(input)
        # print("outputs :", outputs)
        # _, predicted_1 = torch.max(outputs[0].data) # 维度上找到张量中的最大值,_是最大值，predicted 则是最大值所在的索引
        predicted_1 = torch.argmax(outputs, dim=1)
        #print('label[0]', label[0])  # label[0] tensor(1, device='cuda:0')
        total_samples += 1
        # predicted_1是最大值索引，是一个数，用了[0]之后就是一个数，没有影响 
        correct_predictions_1 += (predicted_1[0] == label[0]).sum().item()
        #将正常案例当成正例，即0为正例
        if predicted_1[0] == 0 and label[0] == 0:
            TP += 1
        if predicted_1[0] == 1 and label[0] in [0, 2, 3, 4]:
            FN += 1
        if predicted_1[0] == 2 and label[0] in [0, 1, 3, 4]:
            FN += 1
        if predicted_1[0] == 3 and label[0] in [0, 1, 2 ,4]:
            FN += 1
        if predicted_1[0] == 4 and label[0] in [0, 1, 2, 3]:
            FN += 1  
        if predicted_1[0] == 0 and label[0] in [1, 2, 3, 4]:
            FP += 1

        # print("predicted_1 :", predicted_1)
        # print("label :", label)
        # print("correct_predictions :", correct_predictions)

    accuracy_1 = correct_predictions_1 / total_samples
    
    return [accuracy_1, TP, FN, FP]

###########################################################################
def test_single_volume_change_skip_concat(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input, True), dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list



def test_view_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)      
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0) # zoom : resize to 256,256
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out_main,_,_ = net(input,input,input)
                out = torch.argmax(torch.softmax(
                    out_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list, prediction
    
def test_view_single_volume_single_input(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)      
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0) # zoom : resize to 256,256
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out_main = net(input)
                out = torch.argmax(torch.softmax(
                    out_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list, prediction


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main, _, _, _ = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_cct(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_feature_away(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out1, feature1 = net(input)
                out = torch.argmax(torch.softmax(
                    out1, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

