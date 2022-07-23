from utils.utils import *


def log(opt, mode, inputs, outputs, losses,step,writers):
    """Write an event to the tensorboard events file
    """
    writer = writers[mode]
    for l, v in losses.items():
        writer.add_scalar("{}".format(l), v, step)

    for j in range(min(2, opt.batch_size)):  # write a maxmimum of four images
        for s in range(1):
            for frame_id in opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, step)
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred0_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s,0)][j].data, step)
            writer.add_image(
                "disp_residual0_{}/{}".format(s, j),
                normalize_image(outputs[("disp", 0, s,0)][j]), step)

            if opt.training_stage == 2:
                writer.add_image(
                    "color_teacher_{}_{}/{}".format(frame_id, s, j),
                    outputs[("color_teacher", frame_id, 0)][j].data,
                    "disp_select_{}/{}".format(s, j),
                    normalize_image((outputs["teacher_repro"][j][None, ...])), step)

            if not opt.disable_automasking:
                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], step)

            if opt.use_geometric_mask:
                writer.add_image(
                    "reconstruct_disp_-1_{}/{}".format(s, j),
                    normalize_image(outputs[("depth_warp", -1,0)][j]), step)
                writer.add_image(
                    "reconstruct_disp_1_{}/{}".format(s, j),
                    normalize_image(outputs[("depth_warp", 1, 0)][j]), step)
                writer.add_image(
                    "reconstruct_mask_{}/{}".format(s, j),
                    normalize_image(outputs[("valid_mask", 0)][j]), step)
