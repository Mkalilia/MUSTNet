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
### \[One stage\] Self-supervised manner
Firstly, you can train our model in an fully self-supervised manner with 
~~~
python train.py --training_stage 1 --set_assistant_teacher False
~~~
Then, you can train the model with only one coefficient decoder.
### \[Two stages\] The first training stage for the teachers
~~~
python train.py --training_stage 1 --num_teacher 4 --set_assistant_teacher True
~~~
Then, you can train the model with 4 coefficient decoder.
## KITTI Evaluation
Firstly, prepare the ground truth depth maps by running:
~~~
python export_gt_depth.py --data_path ./kitti_RAW
~~~
Then put the pretrained Models in ./models

You can evaluate the pretrained models by running:
~~~
python evaluate_depth.py --data_path ./kitti_RAW --load_weights_folder ./models/MUSTNet_S_640x192 --MUSTNet
python evaluate_depth.py --data_path ./kitti_RAW --load_weights_folder ./models/MUSTNet_S_124x320 --MUSTNet
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
We also provide pre-computed depth maps for supervised training and evaluation:

## References
