from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.utils import readlines
from option.options import MustNetOptions
import core.datasets as datasets
import networks.muse_net as networks
from core.compute_result import generate_images_pred
from core.compute_losses import compute_reprojection_loss
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "utils/splits")

def inference_depth(opt):
    """
    Inference the teacher pseudo labels
    """
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "train_files.txt"))

    # load the datasets
    dataset = datasets.KITTIRAWDataset(opt.data_path,None,None,filenames,
                                       opt.height, opt.width,
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=False, drop_last=False)

    # load encoder/decoder
    encoder = networks.ResnetEncoder(opt.num_layers, True)
    encoder_path = os.path.join(opt.load_weights_folder, "encoder_72000.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth_72000.pth")
    encoder_dict = torch.load(encoder_path)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        opt.width, opt.height))
    i = 0

    save_path = opt.colored_inference_path
    save_np = opt.teacher_path

    path_1, path_2 = filenames[i].split("/")
    path_2_1 = path_2[:26]
    path_2_3 = path_2[27:]
    path_2_2 = path_2_3[0:-2]
    path_2_4 = path_2_3[-1]
    print(filenames[i])

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()
            print(i)

            output = depth_decoder(encoder(input_color),0)

            """generate final mask"""
            generate_images_pred(opt, data, output)
            all_reprojection_error = []
            all_disp = []
            final_disp = torch.zeros(opt.batch_size, 1,opt.height,opt.width).cuda()
            for T_idx in range(opt.num_teacher):
                all_disp.append(output[("disp", 0, 0, T_idx)])
                reprojection_error = []
                for frame_id in opt.frame_ids[1:]:
                    target = data[("color", 0, 0)]
                    pred = output[("color", frame_id, 0, T_idx)]
                    reprojection_error.append(compute_reprojection_loss(pred, target))
                reprojection_error = torch.cat(reprojection_error, 1)
                reprojection_min_error, idxs = torch.min(reprojection_error, dim=1)
                all_reprojection_error.append(reprojection_min_error)
            all_reprojection_error = torch.cat(all_reprojection_error, 1)
            all_reprojection_error, idxs = torch.min(all_reprojection_error, dim=1)
            for ii in range(final_disp.size(0)):
                for jj in range(final_disp.size(-2)):
                    for kk in range(final_disp.size(-1)):
                        final_disp[ii, 0, jj, kk] = all_disp[ii, idxs[ii, jj, kk], jj, kk]

            pred_disp = final_disp
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disp1 = pred_disp[0]
            pred_disp = cv2.resize(pred_disp1, (1280, 384))
            save_pred_disp = pred_disp

            """save colored depth"""
            if save_path  != None:
                vmax = np.percentile(pred_disp, 95)
                normalizer = mpl.colors.Normalize(vmin=pred_disp.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(pred_disp)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)

                if path_2_4 == "r": path_2_5 = "image_03"
                if path_2_4 == "l": path_2_5 = "image_02"
                i += 1
                save_path_final = save_path + path_1 + "/"+ path_2_1 +"/"+path_2_5
                save_path_final_1 = save_path + path_1
                save_path_final_2 = save_path + path_1 + "/" + path_2_1

                if not os.path.exists(save_path_final_1):   os.mkdir(save_path_final_1)
                if not os.path.exists(save_path_final_2):   os.mkdir(save_path_final_2)
                if not os.path.exists(save_path_final):     os.mkdir(save_path_final)

                image_root = os.path.join(save_path_final, "{}_disp.jpeg".format(path_2_2.zfill(10)))
                im.save(image_root)

            """save the npy file"""
            save_path_final_np = save_np + path_1 + "/" + path_2_1 + "/" + path_2_5
            save_path_final_1_np = save_np + path_1
            save_path_final_2_np = save_np + path_1 + "/" + path_2_1

            if not os.path.exists(save_path_final_1_np):  os.mkdir(save_path_final_1_np)
            if not os.path.exists(save_path_final_2_np):  os.mkdir(save_path_final_2_np)
            if not os.path.exists(save_path_final_np):    os.mkdir(save_path_final_np)

            image_numpy_root = os.path.join(save_path_final_np, "{}_disp.npz".format(path_2_2.zfill(10)))
            np.savez(image_numpy_root,save_pred_disp)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    print("-> Inference")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MustNetOptions()
    inference_depth(options.parse())
