# JointBDOE
Code for my paper "Joint Multi-Person Body Detection and Orientation Estimation via One Unified Embedding"

`Human body orientation estimation (HBOE) is widely applied into various applications, including robotics, surveillance, pedestrian analysis and autonomous driving. Although many approaches have been addressing the HBOE problem from specific under-controlled scenes to challenging in-the-wild environments, they assume human instances are already detected and take a well cropped sub-image as the input. This setting is less efficient and prone to errors in real application, such as crowds of people. In the paper, we propose a single-stage end-to-end trainable framework for tackling the HBOE problem with multi-persons. By integrating the prediction of bounding boxes and direction angles in one embedding, our method can jointly estimate the location and orientation of all bodies in one image directly. Our key idea is to integrate the HBOE task into the multi-scale anchor channel predictions of persons for concurrently benefiting from engaged intermediate features. Therefore, our approach can naturally adapt to difficult instances involving low resolution and occlusion as in object detection. We validated the efficiency and effectiveness of our method in the recently presented benchmark MEBOW with extensive experiments. Besides, we completed ambiguous instances ignored by the MEBOW dataset, and provided corresponding weak body-orientation labels to keep the integrity and consistency of it for supporting studies toward multi-persons.`

## Installation


## Dataset Preparing


## Training and Testing


## References
