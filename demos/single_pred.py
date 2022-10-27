

import os
import cv2
import argparse
import json
import copy
import numpy as np

from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config

from utils.transforms import get_affine_transform

import models


os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def preprocessing(cfg, imgfile, bbox, img_transforms):

    # config parameters
    image_width = cfg.MODEL.IMAGE_SIZE[0]
    image_height = cfg.MODEL.IMAGE_SIZE[1]
    image_size = np.array(cfg.MODEL.IMAGE_SIZE)  # default (192, 256)
    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    # get [center, scale]
    x, y, w, h = bbox[:4]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    # crop person by bbox from image
    data_numpy = cv2.imread(imgfile, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    bbox_crop = data_numpy[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    
    trans = get_affine_transform(center, scale, 0, image_size)
    input = cv2.warpAffine(data_numpy, trans, (image_width, image_height), flags=cv2.INTER_LINEAR)
    input = img_transforms(input)
    input = input.float()

    return input, bbox_crop

def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict
    
def get_anno_img_pairs(imgs_root_path, anno_json_path_COCO, anno_json_path_MEBOW):
    # Processing MEBOW annotations
    print("Processing MEBOW annotations ...")
    img_anno_dict = {}
    anno_dict = json.load(open(anno_json_path_MEBOW, "r"))
    for anno_key, anno_val in tqdm(anno_dict.items()):
        img_id, instance_id = anno_key.split("_")
        img_path = os.path.join(imgs_root_path, img_id.zfill(12) + ".jpg")
        if img_id not in img_anno_dict:
            img_anno_dict[img_id] = {}
        img_anno_dict[img_id][instance_id] = anno_val
     
     
    # Processing COCO annotations
    print("Processing COCO annotations ...")
    imgs_annos_new_dict = {}
    box_with_orientation, box_without_orientation = 0, 0
    coco_dict = json.load(open(anno_json_path_COCO, "r"))
    
    anno_new_full_dict = copy.deepcopy(coco_dict)
    anno_new_full_dict['images'] = []
    anno_new_full_dict['annotations'] = []
    
    imgs_coco_dict = coco_dict['images']
    annos_coco_dict = coco_dict['annotations']
    images_labels_dict = sort_labels_by_image_id(annos_coco_dict)
    for img_coco_dict in tqdm(imgs_coco_dict):
        image_id = str(img_coco_dict['id'])
        if image_id not in img_anno_dict:
            continue  # this image in COCO is not annotated by MEBOW

        anno_COCO_list = images_labels_dict[image_id]
        anno_MEBOW_dict = img_anno_dict[image_id]
        
        ''' format of an instance_id in COCO
        anno_coco_instance= {
            "keypoints": keypoints,
            "num_keypoints": num_keypoints,
            "bbox": bbox,
            'image_id': image_id,
            'id': instance_id,
            'category_id': 1,
            'iscrowd': 0,
            'segmentation': [],
            'area': round(bbox[-1] * bbox[-2], 4),
        }
        '''
        left_instance_list = []  # with body_orientation
        lost_instance_list = []  # without body_orientation
        for anno_coco_instance in anno_COCO_list:
            instance_id = str(anno_coco_instance['id'])
            if instance_id not in anno_MEBOW_dict:
                if anno_coco_instance['iscrowd']:  # give up this crowd instance
                    continue
                # if anno_coco_instance['num_keypoints'] < 3:  # give up this instance
                    # continue
                lost_instance_list.append(anno_coco_instance)
            else:
                body_orientation = anno_MEBOW_dict[instance_id]
                anno_coco_instance['orientation'] = body_orientation
                left_instance_list.append(anno_coco_instance)
                
        assert len(left_instance_list) != 0, "Every image has at least one annotation by MEBOW! --> "+image_id
        
        img_path = os.path.join(imgs_root_path, image_id.zfill(12) + ".jpg")
        imgs_annos_new_dict[img_path] = left_instance_list + lost_instance_list  # all bboxes list
        box_with_orientation += len(left_instance_list)
        box_without_orientation += len(lost_instance_list)
        
        anno_new_full_dict['images'].append(img_coco_dict)
        
        
    print("coco box_with_orientation: ", box_with_orientation)
    print("coco box_without_orientation: ", box_without_orientation)
    
    return imgs_annos_new_dict, anno_new_full_dict


def main(model_state_file, imgs_root_path, anno_json_path_COCO, anno_json_path_MEBOW, debug):
    args = parse_args()
    update_config(cfg, args)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    img_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    print("Model weights loading ...")
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    model.load_state_dict(torch.load(model_state_file))
    model.to(torch.device('cuda:0'))
    
    # switch to evaluate mode
    model.eval()

    imgs_annos_new_dict, anno_new_full_dict = get_anno_img_pairs(
        imgs_root_path, anno_json_path_COCO, anno_json_path_MEBOW)
        
    img_index = 0    
    # for img_path, annos_list in imgs_annos_new_dict.items():
    for img_path, annos_list in tqdm(imgs_annos_new_dict.items()):
        for person_id, anno in enumerate(annos_list):
            bbox = anno['bbox']
            if 'orientation' in anno:            
                orientation_gt = anno['orientation']
                anno['weaklabel'] = 0
            else:
                input, bbox_crop = preprocessing(cfg, img_path, bbox, img_transforms)
                input_batch = torch.unsqueeze(input, dim=0)
                plane_output, hoe_output = model(input_batch.cuda())
                orientation_72_neuron = hoe_output.detach().cpu().numpy()
                orientation_pred = np.argmax(orientation_72_neuron[0]) * 5

                anno['orientation'] = float(orientation_pred)  # add weak label for unlabeld person instance
                orientation_gt = "#"
                anno['weaklabel'] = 1
                
            if debug:
                if anno['weaklabel'] == 0:
                    input, bbox_crop = preprocessing(cfg, img_path, bbox, img_transforms)
                    input_batch = torch.unsqueeze(input, dim=0)
                    plane_output, hoe_output = model(input_batch.cuda())
                    orientation_72_neuron = hoe_output.detach().cpu().numpy()
                    orientation_pred = np.argmax(orientation_72_neuron[0]) * 5
                    
                # print(img_index, img_path, person_id, "\t", orientation_pred, "\t", orientation_gt)
                if not os.path.exists("temp"): os.mkdir("temp")
                cv2.imwrite(os.path.join("temp", str(img_index) + "-" + os.path.split(img_path)[-1][:-4] + 
                    "-" + str(person_id) + "-" + str(orientation_pred) + "-" + str(orientation_gt) + ".jpg"), 
                    bbox_crop)
            
            anno_new_full_dict['annotations'].append(anno)
            
        img_index += 1
        if debug and img_index > 30: break
    
    return anno_new_full_dict
    
                
if __name__ == '__main__':
    
    debug = False  # True or False
    model_state_file = "weights/model_hboe.pth"

    '''(remote ubuntu server) The original COCO dataset'''
    imgs_COCO_train = "/datasdc/zhouhuayi/dataset/coco/images/train2017"  # 118287 images, 262465 instances
    imgs_COCO_val = "/datasdc/zhouhuayi/dataset/coco/images/val2017"  # 5000 images, 11004 instances
    anno_COCO_train = "/datasdc/zhouhuayi/dataset/coco/annotations/person_keypoints_train2017.json"
    anno_COCO_val = "/datasdc/zhouhuayi/dataset/coco/annotations/person_keypoints_val2017.json"

    '''(remote ubuntu server)'''
    anno_MEBOW_train = "/datasdc/zhouhuayi/dataset/coco/annotations_MEBOW/train_hoe.json"  # 51836 images, 127844 instances
    anno_MEBOW_val = "/datasdc/zhouhuayi/dataset/coco/annotations_MEBOW/val_hoe.json"  # 2171 images, 5536 instances
   
    save_anno_full_train = "/datasdc/zhouhuayi/dataset/coco/annotations_JointBDOE/JointBDOE_coco_weaklabel_train.json"
    save_anno_full_val = "/datasdc/zhouhuayi/dataset/coco/annotations_JointBDOE/JointBDOE_coco_weaklabel_val.json"


    anno_new_full_dict_val = main(model_state_file, imgs_COCO_val, anno_COCO_val, anno_MEBOW_val, debug)
    print("[val-set] The images/instances number in new full annotation with weaklabel: %d / %d"%(
        len(anno_new_full_dict_val['images']), len(anno_new_full_dict_val['annotations'])))
    with open(save_anno_full_val, "w") as dst_ann_file:
        json.dump(anno_new_full_dict_val, dst_ann_file)
        
    anno_new_full_dict_train = main(model_state_file, imgs_COCO_train, anno_COCO_train, anno_MEBOW_train, debug)
    print("[train-set] The images/instances number in new full annotation with weaklabel: %d / %d"%(
        len(anno_new_full_dict_train['images']), len(anno_new_full_dict_train['annotations'])))
    with open(save_anno_full_train, "w") as dst_ann_file:
        json.dump(anno_new_full_dict_train, dst_ann_file) 
    
'''
python tools/single_pred.py --cfg experiments/coco/segm-4_lr1e-3.yaml    

'''