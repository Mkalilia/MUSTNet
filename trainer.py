from __future__ import absolute_import, division, print_function

import time
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from tensorboardX import SummaryWriter
from utils.utils import *
from utils.log_tensorboard import *
from utils.kitti_utils import *
from utils.generate_mask import *

import core.datasets as datasets
from core.compute_result import *
from core.compute_losses import *
from core.compute_losses_st2 import *
import networks.muse_net as networks


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # the depth model
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales,1,self.opt.num_teacher)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # the pose model
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        datasets_dict = {"kitti": datasets.KITTIRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        # the dataset splits
        fpath = os.path.join(os.path.dirname(__file__), "utils/splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path,None,None,train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path,None,None, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        self.save_opts()

    def set_train(self):
        """
        Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """
        Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """
        Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch(self.epoch)
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self,epoch):
        """
        Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        print("Training")
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(self.models,inputs,epoch)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 500 == 0
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                log(self.opt,"train", inputs, outputs, losses,self.step,self.writers)
                self.val(epoch)
            self.step += 1

    def process_batch(self, models, inputs,epoch):
        """
        Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.opt.device)

        if self.opt.distill_train_mode == "depth":
            features = models["encoder"](inputs["color_aug", 0, 0])
            outputs = models["depth"](features, 0)
        if self.opt.use_geometric_mask:
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                features = models["encoder"](inputs["color_aug", frame_id, 0])
                outputs.update(models["depth"](features, frame_id, None))

        outputs.update(predict_poses(self.opt, models, inputs))

        if self.opt.training_stage ==2:
            if self.opt.distill_train_mode == "depth":
                batch_size, depth, height, width = outputs[("disp", 0, 0)].shape
            else:
                batch_size, _, height, width = self.opt.batch_size, 1, self.opt.height, self.opt.width
            grid = torch.linspace(
                1e-3, 1, 64,
                requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            grid = grid.repeat(batch_size, 1, height, width).float()
            outputs.update(match_features(self.opt, inputs, outputs, grid))

        generate_images_pred(self.opt, inputs, outputs)
        if self.opt.training_stage == 1:
            if (self.opt.set_assistant_teacher and epoch >=self.opt.use_assistant_epoch):
                outputs[("weights_0")] = outputs["weights_0"].detach()
            losses = compute_losses_S1(self.opt, inputs, outputs,epoch)
        else:
            losses = compute_losses_S2(self.opt, inputs, outputs)
        if self.opt.use_geometric_mask:
            generate_depth_pred(self.opt, outputs)
            losses.update(compute_pairwise_loss(self.opt, outputs, losses))

        return outputs, losses

    def val(self,epoch):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(self.models,inputs,epoch)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            log(self.opt,"val", inputs, outputs, losses,self.step,self.writers)
            del inputs, outputs, losses

        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """
        Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def save_opts(self):
        """
        Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """
        Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        print("saved model.")

    def load_model(self):
        """
        Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            if not self.opt.use_weight_teacher:
                path = os.path.join(self.opt.load_weights_folder, "{}_72000.pth".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
            else:
                path = os.path.join(self.opt.load_weights_folder, "{}_72000.pth".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
