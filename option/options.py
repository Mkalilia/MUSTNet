
from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = "/home/kalilia/KITTI_odometry/KITTI_odometry/KITTI_monodepth2" # the directory that options.py resides in
teacher_dir = "/home/kalilia/KITTI_dataset/average_depth1/npy"
teacher_mask_dir = "/home/kalilia/KITTI_dataset/depth_teacher_0.109/npy_weight"
flow_dir = "/home/kalilia/KITTI_dataset/optical_flow"
class MustNetOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data (Img)",
                                 default=file_dir)
        self.parser.add_argument("--teacher_path",
                                 help="path to the teacher prediction (npy) for the 2nd training stage",
                                 default=None)
        self.parser.add_argument("--flow_path",
                                 help="(Optional) the pre-predicted depth for the dynamic mask",
                                 default=None)
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="/home/kalilia/nyud_datset2/eccv_save")
        # TRAINING options
        self.parser.add_argument("--device",
                                 help="step size of the scheduler",
                                 default="cuda")
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="1026_POSE_add_geometric_loss")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default = 640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=6)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        # Inference options
        self.parser.add_argument("--colored_inference_path",
                                 help="path to the teacher prediction (npy) for the 2nd training stage",
                                 default=None)

        # ABLATION options
        self.parser.add_argument("--use_geometric_mask",
                                 help="use the geometric masks generated by adjacent frames",
                                 default=False)
        self.parser.add_argument("--set_assistant_teacher",
                                 help="if fix the main teacher and only train the assistant teacher after 10 epochs",
                                 default=False)
        self.parser.add_argument("--use_assistant_epoch",
                                 type = int,
                                 help="if fix the main teacher and only train the assistant teacher after 10 epochs",
                                 default=15)
        self.parser.add_argument("--use_flow",
                                 help="if  use the optical flow generated by raft to generate dynamic masks",
                                 default=False)
        self.parser.add_argument("--num_teacher",
                                 help="num of teachers, if set None, train the student model in unsupervised manner",
                                 default=None)
        self.parser.add_argument("--training_stage",
                                 type=int,
                                 help="the traning stage of our work",
                                 default=1,
                                 choices=[1, 2])
        self.parser.add_argument("--use_teacher_mask",
                                 help="step size of the scheduler",
                                 default=False)
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 default=False)
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)
        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 default=None)
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["pose","pose_encoder","depth","depth_encoder"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=20)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        # EVALUATION options
        self.parser.add_argument("--save_weights",
                                 help="if set evaluates in stereo mode",
                                 default=False)
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 default = False)
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 default= True)
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen_benchmark",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10","nuscenes"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 default=False)
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 default=False)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
