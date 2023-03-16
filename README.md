# JointBDOE
Code for my paper [Joint Multi-Person Body Detection and Orientation Estimation via One Unified Embedding](https://arxiv.org/abs/2210.15586)

* [**2023-03-16**] We released our JointBDOE-S/M/L pretrained models on COCO-MEBOW in [Hugging Face](https://huggingface.co/HoyerChou/JointBDOE). Please follow the [[Training and Testing](#training-and-testing)] section to test on your own images/videos.


**Abstract:** Human body orientation estimation (HBOE) is widely applied into various applications, including robotics, surveillance, pedestrian analysis and autonomous driving. Although many approaches have been addressing the HBOE problem from specific under-controlled scenes to challenging in-the-wild environments, they assume human instances are already detected and take a well cropped sub-image as the input. This setting is less efficient and prone to errors in real application, such as crowds of people. In the paper, we propose a single-stage end-to-end trainable framework for tackling the HBOE problem with multi-persons. By integrating the prediction of bounding boxes and direction angles in one embedding, our method can jointly estimate the location and orientation of all bodies in one image directly. Our key idea is to integrate the HBOE task into the multi-scale anchor channel predictions of persons for concurrently benefiting from engaged intermediate features. Therefore, our approach can naturally adapt to difficult instances involving low resolution and occlusion as in object detection. We validated the efficiency and effectiveness of our method in the recently presented benchmark MEBOW with extensive experiments. Besides, we completed ambiguous instances ignored by the MEBOW dataset, and provided corresponding weak body-orientation labels to keep the integrity and consistency of it for supporting studies toward multi-persons.

<table>
<tr>
<th>Examples of COCO-MEBOW with full labels</th>
<th>The overall architecture of JointBDOE</th>
</tr>
<tr>
<td><img src="./images/examples.png" width="540"></td>
<td><img src="./images/architecture.png" width="500"></td> 
</tr>
</table>

## Installation

* **Environment:** Anaconda, Python3.8, PyTorch1.10.0(CUDA11.2), wandb

  ```bash
  $ git clone https://github.com/hnuzhy/JointBDOE.git
  $ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

  # Codes are only evaluated on GTX3090 + CUDA11.2 + PyTorch1.10.0.
  $ pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html
  ```

## Dataset Preparing

* **Original MEBOW (CVPR2020) for singel-person BOE:** Monocular Estimation of Body Orientation In the Wild [[project link](https://chenyanwu.github.io/MEBOW/)]. For the images, please download from [MS-COCO](https://cocodataset.org/#download). For its annotation, you can email [czw390@psu.edu](czw390@psu.edu) to get access to human body orientation annotation. More details can be found in their official [[code in github](https://github.com/ChenyanWu/MEBOW)]

* **Our Full MEBOW for multi-person BOE:** After downloading both the original MEBOW annotations (`train_hoe.json` and `val_hoe.json`) and the COCO person annotations (`person_keypoints_train2017.json` and `person_keypoints_val2017.json`), also all COCO images labeled by MEBOW, you can follow steps below to generate annotations (`JointBDOE_coco_weaklabel_train.json` and `JointBDOE_coco_weaklabel_val.json`) for full MEBOW with weak labels.
  ```bash
  # install and config the MEBOW code project
  $ git clone https://github.com/ChenyanWu/MEBOW

  # copy our generating file single_pred.py under ./demos/ to the MEBOW code project ./tools/
  $ python tools/single_pred.py --cfg experiments/coco/segm-4_lr1e-3.yaml

  # more details can be found in our single_pred.py file
  $ cat demos/single_pred.py 
  ```

## Training and Testing

* **Yaml:** Please refer the [JointBDOE_weaklabel_coco.yaml](./data/JointBDOE_weaklabel_coco.yaml) file to config your own .yaml file

* **Pretrained weights:** 
  ```
  # For YOLOv5 weights, please download the version 5.0 that we have used
  yolov5s6.pt
  [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s6.pt]
  yolov5m6.pt
  [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m6.pt]
  yolov5l6.pt
  [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l6.pt]
  
  # For JointBDOE weights, we currently release all the yolov5-based models in Hugging Face.
  # Below are its evaluation results
  
  coco_s_1024_e500_t020_w005_best.pt
  Left bbox number (MEBOW): 5511 / 5536;
  [WMAE,Acc-05,Acc-15,Acc-22.5,Acc-30,Acc-45]: 13.1296,44.0755,79.8948,88.0784,91.7982,94.2297
  Left bbox number (FULL): 8821 / 9059;
  [WMAE,Acc-05,Acc-15,Acc-22.5,Acc-30,Acc-45]: 20.6699,37.048,69.6973,78.9593,83.5053,87.8925

  coco_m_1024_e500_t020_w005_best.pt
  Left bbox number (MEBOW): 5508 / 5536;
  [WMAE,Acc-05,Acc-15,Acc-22.5,Acc-30,Acc-45]: 11.9071,46.4052,83.0065,89.724,92.8649,95.4248
  Left bbox number (FULL): 8844 / 9059;
  [WMAE,Acc-05,Acc-15,Acc-22.5,Acc-30,Acc-45]: 18.7632,39.8349,73.5414,81.3659,85.5043,89.5183
  
  coco_l_1024_e500_t020_w005_best.pt
  Left bbox number (MEBOW): 5506 / 5536;
  [WMAE,Acc-05,Acc-15,Acc-22.5,Acc-30,Acc-45]: 11.2068,47.53,83.3636,90.5013,93.7704,96.0407
  Left bbox number (FULL): 8839 / 9059;
  [WMAE,Acc-05,Acc-15,Acc-22.5,Acc-30,Acc-45]: 17.8479,40.7512,74.4315,82.2944,86.5369,90.4175
  ```
  
* **Training:**

  Training `yolov5s6` based JointBDOE model 500 epochs on 4 GTX-3090 GPUs with batchsize 180
  ```bash
  python -m torch.distributed.launch --nproc_per_node 4 \
      train.py --workers 20 --device 0,1,2,4 \
      --img 1024 --batch 180 --epochs 500 \
      --data data/JointBDOE_weaklabel_coco.yaml --hyp data/hyp-p6.yaml \
      --val-scales 1 --val-flips -1 \
      --weights weights/yolov5s6.pt --project runs/JointBDOE \
      --mse_conf_thre 0.20 --mse_loss_w 0.05 --name coco_s_1024_e500_t020_w005
  ```
  Training `yolov5m6` based JointBDOE model 500 epochs on 4 GTX-3090 GPUs with batchsize 96
  ```bash
  python -m torch.distributed.launch --nproc_per_node 4 \
      train.py --workers 20 --device 0,1,2,4 \
      --img 1024 --batch 96 --epochs 500 \
      --data data/JointBDOE_weaklabel_coco.yaml --hyp data/hyp-p6.yaml \
      --val-scales 1 --val-flips -1 \
      --weights weights/yolov5m6.pt --project runs/JointBDOE \
      --mse_conf_thre 0.20 --mse_loss_w 0.05 --name coco_m_1024_e500_t020_w005
  ```
  Training `yolov5l6` based JointBDOE model 500 epochs on 4 GTX-3090 GPUs with batchsize 48
  ```bash
  python -m torch.distributed.launch --nproc_per_node 4 train.py \
      --workers 20 --device 0,1,2,4 \
      --img 1024 --batch 48 --epochs 500 \
      --data data/JointBDOE_weaklabel_coco.yaml --hyp data/hyp-p6.yaml \
      --val-scales 1 --val-flips -1 \
      --weights weights/yolov5l6.pt --project runs/JointBDOE \
      --mse_conf_thre 0.20 --mse_loss_w 0.05 --name coco_l_1024_e500_t020_w005
  ```
  
* **Testing:**

  For evaluation on the val-set of full MEBOW, e.g. testing the trained `coco_s_1024_e500_t020_w005` project.
  ```bash
  python val.py --rect --data data/JointBDOE_weaklabel_coco.yaml --img 1024 \
      --weights runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt --batch-size 16 --device 3
  ```
  For testing one single image with multi-persons.
  ```bash
  # [COCO][JointBDOE - YOLOv5S] 
  python demos/image.py --weights runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt \
      --device 3 --img-path test_imgs/COCO/ --conf-thres 0.3 --iou-thres 0.5 --gt-show --thickness 1
      
  # [CrowdHuman][JointBDOE - YOLOv5S] 
  python demos/image.py --weights runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt \
      --device 3 --img-path test_imgs/CrowdHuman/ --conf-thres 0.3 --iou-thres 0.5
  ```
  
## References

* [YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
* [Code for "MEBOW: Monocular Estimation of Body Orientation In the Wild"](https://github.com/ChenyanWu/MEBOW)

## Citation

If you use our dataset or models in your research, please cite with:
```
@inproceedings{wu2020mebow,
  title={MEBOW: Monocular Estimation of Body Orientation In the Wild},
  author={Wu, Chenyan and Chen, Yukun and Luo, Jiajia and Su, Che-Chun and Dawane, Anuja and Hanzra, Bikramjot and Deng, Zhuo and Liu, Bilan and Wang, James Z and Kuo, Cheng-hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3451--3461},
  year={2020}
}

@article{zhou2022joint,
  title={Joint Multi-Person Body Detection and Orientation Estimation via One Unified Embedding},
  author={Zhou, Huayi and Jiang, Fei and Si, Jiaxin and Lu, Hongtao},
  journal={arXiv preprint arXiv.2210.15586},
  year={2022}
}
```
