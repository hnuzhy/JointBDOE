
import os
import json
import numpy as np
import copy
from tqdm import tqdm

def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict


def calculate_bbox_iou(bboxA, bboxB, format='xyxy'):
    if format == 'xywh':  # xy is in top-left, wh is size
        [Ax, Ay, Aw, Ah] = bboxA[0:4]
        [Ax0, Ay0, Ax1, Ay1] = [Ax, Ay, Ax+Aw, Ay+Ah]
        [Bx, By, Bw, Bh] = bboxB[0:4]
        [Bx0, By0, Bx1, By1] = [Bx, By, Bx+Bw, By+Bh]
    if format == 'xyxy':
        [Ax0, Ay0, Ax1, Ay1] = bboxA[0:4]
        [Bx0, By0, Bx1, By1] = bboxB[0:4]
        
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)

def mean_absolute_error_calculate(gt_json_path, pd_json_path):
    matched_iou_threshold = 0.5
    score_threshold = 0.5

    gt_data, pd_data = [], []  # shapes of both should be N*1
    gt_data_MEBOW, pd_data_MEBOW = [], []  # shapes of both should be N*1, original labels in the MEBOW dataset

    gt_json = json.load(open(gt_json_path, "r"))
    pd_json = json.load(open(pd_json_path, "r"))
    
    gt_labels_list = gt_json['annotations']
    pd_images_labels_dict = sort_labels_by_image_id(pd_json)

    for gt_label_dict in tqdm(gt_labels_list):  # matching for each GT label
        image_id = str(gt_label_dict['image_id'])
        gt_bbox = gt_label_dict['bbox']
        gt_angle = gt_label_dict['orientation']
        
        if image_id not in pd_images_labels_dict:  # this image has no bboxes been detected
            continue
            
        pd_results = pd_images_labels_dict[image_id]
        max_iou, matched_index = 0, -1
        for i, pd_result in enumerate(pd_results):  # match predicted bboxes in target image
            score = pd_result['score']
            # if score < score_threshold:
                # continue
            pd_bbox = pd_result['bbox']
            temp_iou = calculate_bbox_iou(pd_bbox, gt_bbox, format='xywh')
            if temp_iou > max_iou:
                max_iou = temp_iou
                matched_index = i
                
        if max_iou > matched_iou_threshold:
            pd_angle = pd_results[matched_index]['orientation']
            gt_data.append(gt_angle)
            pd_data.append(pd_angle)
            if 'weaklabel' in gt_json_path:
                if gt_label_dict['weaklabel'] != 1:
                    gt_data_MEBOW.append(gt_angle)
                    pd_data_MEBOW.append(pd_angle)  
                

    if 'weaklabel' in gt_json_path:
        total_num_MEBOW = sum([1-anno['weaklabel'] for anno in gt_labels_list])
        left_num_MEBOW = len(gt_data_MEBOW)
        if left_num_MEBOW == 0:
            print("Left bbox number (MEBOW): %d / %d; [WMAE, Acc-05, Acc-15, Acc-30]: %s, %s, %s, %s"%(
                0, total_num_MEBOW, 90, 0, 0, 0))
        else:
            error_list_MEBOW = np.abs(np.array(gt_data_MEBOW) - np.array(pd_data_MEBOW))
            error_list_MEBOW = np.min((error_list_MEBOW, 360 - error_list_MEBOW), axis=0)  # orientation range is [0, 360)
            wmae_MEBOW = np.mean(error_list_MEBOW, axis=0)
            acc05_MEBOW = np.sum(error_list_MEBOW <= 5) * 100 / left_num_MEBOW
            acc15_MEBOW = np.sum(error_list_MEBOW <= 15) * 100 / left_num_MEBOW
            acc30_MEBOW = np.sum(error_list_MEBOW <= 30) * 100 / left_num_MEBOW
            print("Left bbox number (MEBOW): %d / %d; [WMAE, Acc-05, Acc-15, Acc-30]: %s, %s, %s, %s"%(
                left_num_MEBOW, total_num_MEBOW, 
                round(wmae_MEBOW, 4), round(acc05_MEBOW, 4), round(acc15_MEBOW, 4), round(acc30_MEBOW, 4)))


    total_num = len(gt_labels_list)
    left_num = len(gt_data)
    
    if left_num == 0:
        return total_num, 0, 90, 0, 0, 0

    error_list = np.abs(np.array(gt_data) - np.array(pd_data))
    error_list = np.min((error_list, 360 - error_list), axis=0)  # orientation range is [0, 360)
    wmae = np.mean(error_list, axis=0)

    acc05 = np.sum(error_list <= 5) * 100 / left_num
    acc15 = np.sum(error_list <= 15) * 100 / left_num
    acc30 = np.sum(error_list <= 30) * 100 / left_num
    
    return total_num, left_num, wmae, acc05, acc15, acc30
    
