from networks.layers import *

def predict_poses(opt,models, inputs):
    """
    Predict camera pose of adjacent frames
    """
    outputs = {}
    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in opt.frame_ids}
    for f_i in opt.frame_ids[1:]:
        if f_i < 0:
            pose_inputs = [pose_feats[f_i], pose_feats[0]]
        else:
            pose_inputs = [pose_feats[0], pose_feats[f_i]]
        pose_inputs = [models["pose_encoder"](torch.cat(pose_inputs, 1))]
        axisangle, translation = models["pose"](pose_inputs)
        outputs[("axisangle", 0, f_i)] = axisangle
        outputs[("translation", 0, f_i)] = translation
        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
    return outputs

def generate_images_pred(opt, inputs, outputs):
    """
    Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    backproject_depth = {}
    project_3d = {}
    for scale in opt.scales:
        h = opt.height // (2 ** scale)
        w = opt.width // (2 ** scale)
        backproject_depth[scale] = BackprojectDepth(opt.batch_size, h, w)
        backproject_depth[scale].to(opt.device)
        project_3d[scale] = Project3D(opt.batch_size, h, w)
        project_3d[scale].to(opt.device)
    if opt.num_teacher == None:
        for scale in opt.scales:
            disp = outputs[("disp", 0, scale,0)]

            disp = F.interpolate(
                disp, [opt.height, opt.width], mode="bilinear", align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)
            outputs[("depth", 0, scale,0)] = depth

            for i, frame_id in enumerate(opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale,0)] = pix_coords
                outputs[("color", frame_id, scale,0)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale,0)],
                    padding_mode="border")

                if not opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    else:
        for teacher_idx in range(opt.num_teacher):
            for scale in opt.scales:
                disp = outputs[("disp", 0, scale,teacher_idx)]
                disp = F.interpolate(
                    disp, [opt.height, opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
                _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)
                outputs[("depth", 0, scale,teacher_idx)] = depth

                for i, frame_id in enumerate(opt.frame_ids[1:]):
                    T = outputs[("cam_T_cam", 0, frame_id)]
                    cam_points = backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)
                    outputs[("sample", frame_id, scale,teacher_idx)] = pix_coords
                    outputs[("color", frame_id, scale,teacher_idx)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample",  frame_id, scale,teacher_idx)],
                        padding_mode="border")

                    if not opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]


def generate_depth_pred(opt, outputs):
    for scale in opt.scales:
        disp_center = outputs[("disp", 0, scale)]
        disp_center = F.interpolate(
            disp_center, [opt.height, opt.width], mode="bilinear", align_corners=False)
        _, depth_center = disp_to_depth(disp_center, opt.min_depth, opt.max_depth)
        disp_last = outputs[("disp", -1, scale)]
        disp_last = F.interpolate(
            disp_last, [opt.height, opt.width], mode="bilinear", align_corners=False)
        _, depth_last = disp_to_depth(disp_last, opt.min_depth, opt.max_depth)
        disp_next = outputs[("disp", 1, scale)]
        disp_next = F.interpolate(
            disp_next, [opt.height, opt.width], mode="bilinear", align_corners=False)
        _, depth_next = disp_to_depth(disp_next, opt.min_depth, opt.max_depth)

        outputs[("depth", -1, scale)] = depth_last
        outputs[("depth", 1, scale)] = depth_next

        sample_depth01 = outputs[("sample", 1, scale)]
        sample_depth00 = outputs[("sample", -1, scale)]

        outputs[("depth_warp", 1, scale)] = F.grid_sample(
            depth_next,
            sample_depth01,
            padding_mode="zeros")
        outputs[("depth_warp", -1, scale)] = F.grid_sample(
            depth_last,
            sample_depth00,
            padding_mode="zeros")

        edge_mask1 = outputs[("depth_warp", -1, scale)] <= 1e-5
        edge_mask2 = outputs[("depth_warp", 1, scale)] <= 1e-5
        outputs[("edge_mask", scale)] = (1 - torch.max(edge_mask1, edge_mask2).float()).detach()
        valid_points = sample_depth01.abs().max(dim=-1)[0] <= 1
        valid_points0 = sample_depth00.abs().max(dim=-1)[0] <= 1
        valid_points_min = torch.min(valid_points, valid_points0)
        outputs[("valid_mask", scale)] = valid_points_min.unsqueeze(1).float().detach()
    return outputs