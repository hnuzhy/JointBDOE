import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
import argparse
import yaml
import cv2
import math
import os.path as osp
import numpy as np

import os
import json

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load


def plot_bbox_orientation(img_ori, bboxes, orientations, gt_labels, thickness=1):
    img_h, img_w, img_c = img_ori.shape
    # img_vis = np.ones((img_h, img_w, img_c)) * 127  # gray canvas
    img_vis = np.ones((img_h, img_w, img_c)) * 255  # white canvas
    cv2.rectangle(img_vis, (0, 0), (img_w-1, img_h-1), (0,0,0), 2, lineType=cv2.LINE_AA)
    
    if len(gt_labels) != 0:
        gt_orientation_list = match_pred_bbox_with_gt(gt_labels, bboxes)
    
    for index, (bbox, orientation) in enumerate(zip(bboxes, orientations)):
        [x1, y1, x2, y2] = bbox
        x, y, w, h = x1, y1, x2-x1, y2-y1
        
        img_ori = cv2.rectangle(img_ori, (int(x), int(y)), (int(x+w), int(y+h)), 
            (0,0,255), thickness=thickness, lineType=cv2.LINE_AA) # red
        img_vis = cv2.rectangle(img_vis, (int(x), int(y)), (int(x+w), int(y+h)), 
            (0,255,0), thickness=thickness, lineType=cv2.LINE_AA) # green
        
        arrow_len_1 = int((w + h) / np.pi / 2)
        arrow_len_2 = int(min(w, h) / 3)
        arrow_len = min(arrow_len_1, arrow_len_2)
        end_px = int(x + w/2 - arrow_len * np.sin(orientation/180 * np.pi))
        end_py = int(y + h/2 - arrow_len * np.cos(orientation/180 * np.pi))
        start_px = int(x + w/2)
        start_py = int(y + h/2)
        
        img_ori = cv2.arrowedLine(img_ori, (start_px, start_py), (end_px, end_py), (0,255,255), 
            thickness=thickness, line_type=cv2.LINE_AA, tipLength=0.3)  # yellow for pd
        if len(gt_labels) != 0:
            [orientation_gt, bbox_gt] = gt_orientation_list[index]
            if orientation_gt is not None:
                end_px_gt = int(x + w/2 - arrow_len * np.sin(orientation_gt/180 * np.pi))
                end_py_gt = int(y + h/2 - arrow_len * np.cos(orientation_gt/180 * np.pi))
                img_ori = cv2.arrowedLine(img_ori, (start_px, start_py), (end_px_gt, end_py_gt), (0,255,0), 
                    thickness=thickness, line_type=cv2.LINE_AA, tipLength=0.3)  # green for gt

        img_vis = cv2.circle(img_vis, (start_px, start_py), arrow_len, (0,0,255), 2) # red
        # img_vis = cv2.arrowedLine(img_vis, (start_px, start_py), (end_px, end_py), (255,255,0), 
            # thickness=thickness, line_type=cv2.LINE_AA, tipLength=0.3) # cyan
        img_vis = cv2.line(img_vis, (start_px, start_py), (end_px, end_py), (255,255,0),
            thickness=thickness, lineType=cv2.LINE_AA)
        img_vis = cv2.circle(img_vis, (start_px, start_py), 5, (0,0,0), -1, lineType=cv2.LINE_AA) # black

    return img_ori, img_vis


def calculate_bbox_iou(rectA, rectB):
    # calculate two rectangles IOU(intersection-over-union)
    [Ax0, Ay0, Ax1, Ay1] = rectA[0:4]
    [Bx0, By0, Bx1, By1] = rectB[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)

def match_pred_bbox_with_gt(gt_labels, bboxes):
    gt_orientation_list = []
    for bbox_pd in bboxes:
        [x1, y1, x2, y2] = bbox_pd
        matched_index, max_iou = -1, 0
        for index, gt_label in enumerate(gt_labels):
            bbox_gt_xywh = gt_label["bbox"]
            bbox_gt_xyxy = [ bbox_gt_xywh[0], bbox_gt_xywh[1], 
                bbox_gt_xywh[0]+bbox_gt_xywh[2], bbox_gt_xywh[1]+bbox_gt_xywh[3] ]
            bbox_iou = calculate_bbox_iou(bbox_gt_xyxy, bbox_pd)
            if bbox_iou > max_iou:
                matched_index = index
                max_iou = bbox_iou
        if max_iou > 0.75:  # iou threshold
            bbox_gt_xywh = gt_labels[matched_index]["bbox"]
            bbox_gt_xyxy = [ bbox_gt_xywh[0], bbox_gt_xywh[1], 
                bbox_gt_xywh[0]+bbox_gt_xywh[2], bbox_gt_xywh[1]+bbox_gt_xywh[3] ]
            orientation_gt = gt_labels[matched_index]["orientation"]
            gt_orientation_list.append([orientation_gt, bbox_gt_xyxy])  # bbox_pd matched
        else:
            gt_orientation_list.append([None, None])  # this bbox_pd has not been matched
    return gt_orientation_list

def load_gt_labels_for_COCO_MEBOW():
    anno_full_val = "/datasdc/zhouhuayi/dataset/coco/annotations_JointBDOE/JointBDOE_coco_weaklabel_val.json"
    anno_full_dict = json.load(open(anno_full_val, "r"))
    labels_list = anno_full_dict['annotations']

    # sort_labels_by_image_id
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='test_imgs/100024.jpg', help='path to image or dir')
    parser.add_argument('--data', type=str, default='data/JointBDOE_weaklabel_coco.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)  # 128*8
    parser.add_argument('--weights', default='yolov5m6.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--thickness', type=int, default=2, help='thickness of orientation lines')
    parser.add_argument('--gt-show', action='store_true', help='whether showing GT labels or not')

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.img_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)
    
    if args.gt_show:
        images_labels_dict = load_gt_labels_for_COCO_MEBOW()
        
    print(args.img_path, len(dataset))
    for index in range(len(dataset)):
        
        (single_path, img, im0, _) = next(dataset_iter)
        
        if '_res' in single_path or '_vis' in single_path:
            continue
        
        print(index, single_path, "\n")
        
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out_ori = model(img, augment=True, scales=args.scales)[0]
        out = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, num_angles=data['num_angles'])
        
        # predictions (Array[N, 9]), x1, y1, x2, y2, conf, class, orientation
        bboxes = scale_coords(img.shape[2:], out[0][:, :4], im0.shape[:2]).cpu().numpy()  # native-space pred
        scores = out[0][:, 4].cpu().numpy() 
        orientations = out[0][:, 6:].cpu().numpy() * 360   # N*1, (0,1)*360 --> (0,360)
        
        if args.gt_show:
            img_id = str(int(os.path.split(single_path)[-1][:-4]))
            gt_labels = images_labels_dict[img_id]
        else:
            gt_labels = []

        im0_res, im0_vis = plot_bbox_orientation(im0, bboxes, orientations, gt_labels, thickness=args.thickness)
                
        cv2.imwrite(single_path[:-4]+"_res.jpg", im0_res)
        cv2.imwrite(single_path[:-4]+"_vis.jpg", im0_vis)
        
  