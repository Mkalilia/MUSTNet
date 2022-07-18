# MUSTNet
the source code for MUSTNet

# Useage
## Install
you can install all the dependencies with:
~~~
conda install pytorch=1.2.0 torchvision=0.6.0 -c pytorch
conda install opencv=4.2
pip install scipy=1.4.1
~~~

## Dataset
### KITTI
The KITTI (raw) dataset used in our experiments can be downloaded from the KITTI website.

## Training

## KITTI Evaluation
Firstly, prepare the ground truth depth maps by running:
~~~
python export_gt_depth.py --data_path ./kitti_RAW
~~~
Then put the pretrained Models in 
~~~
python evaluate_depth.py --data_path ./kitti_RAW --load_weights_folder ./models/MUSTNet_K_S_640x192 --MUSTNet
python evaluate_depth.py --data_path ./kitti_RAW --load_weights_folder ./models/HR_Depth_K_S_1280x384 --MUSTNet
~~~
## Pretrained Models
We provided pretrained model as follow:

|Model name|Resolution|Train Dataset|supervision|Abs Rel|delta<1.25|
|:------|:------|:------|:------|:------|:------|
|||||||
|||||||
|||||||
|||||||
## Precomputed Depth Maps

## References
