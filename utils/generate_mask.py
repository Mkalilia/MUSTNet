import torch
from networks.layers import *

def match_features(opt,inputs, outputs, grid):
    with torch.no_grad():
        grid_depth = grid.cuda()
        if opt.distill_train_mode == "depth":
            batch_size, _, height, width = outputs[("disp", 0, 0)].shape
        else:
            batch_size, _, height, width = opt.batch_size, 1, opt.height, opt.width
        num_bins = 64
        lookup_feat1 = inputs[("color", 1, 0)].unsqueeze(1).repeat(1, 64, 1, 1, 1)
        lookup_feat_1 = inputs[("color", -1, 0)].unsqueeze(1).repeat(1, 64, 1, 1, 1)
        curr_feat = inputs[("color", 0, 0)].unsqueeze(1).repeat(1, 64, 1, 1, 1)
        backprojector = BackprojectDepth(batch_size=num_bins,
                                         height=height,
                                         width=width).cuda()
        backprojectorT = BackprojectDepth(batch_size=1,
                                          height=height,
                                          width=width).cuda()
        projector = Project3D(batch_size=num_bins,
                              height=height,
                              width=width).cuda()
        projectorT = Project3D(batch_size=1,
                               height=height,
                               width=width).cuda()
        volume_mask = []
        volume_error = []
        for jj in range(batch_size):
            mask_all_frame = []
            cost_all_frame = []
            te_cost_all_frame = []
            for i, frame_id in enumerate(opt.frame_ids[1:]):

                T_1 = outputs[("cam_T_cam", 0, frame_id)]
                _, depth = disp_to_depth(grid_depth[jj], opt.min_depth, opt.max_depth)
                volume_cam_points = backprojector(
                    depth, inputs[("inv_K", 0)][jj].unsqueeze(0))
                pix_locs = projector(
                    volume_cam_points, inputs[("K", 0)][jj].unsqueeze(0), T_1[jj].unsqueeze(0))
                if frame_id == 1:
                    warped = F.grid_sample(lookup_feat1[jj], pix_locs, padding_mode='zeros', mode='bilinear')
                else:
                    warped = F.grid_sample(lookup_feat_1[jj], pix_locs, padding_mode='zeros', mode='bilinear')
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                        width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (height - 1)
                edge_mask = (x_vals >= 2.0) * (x_vals <= width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= height - 2)
                edge_mask = edge_mask.float()
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask
                diffs = torch.abs(warped - curr_feat[jj]).mean(1)
                diffs = diffs * (edge_mask) + \
                        (diffs * (edge_mask)).max(0)[0].unsqueeze(0) * (1 - edge_mask)
                # probability mask
                prob_sof = torch.softmax(1 / (diffs * 20 + 1e-5), dim=0)
                kk, _ = torch.max(prob_sof, dim=0)
                prob_mask = kk.clone()
                arange_b = 0.6
                prob_mask[prob_mask >= arange_b] = 1
                prob_mask[prob_mask < arange_b] = 0

                photometric_mask = prob_mask
                photometric_mask = photometric_mask.detach()
                mask_new = photometric_mask
                mask_new = mask_new.unsqueeze(0)
                mask_all_frame.append(mask_new)
                # depth teacher error
                depth_teacher_input = inputs[("depth_p"), 0][0, :].unsqueeze(0)
                _, depth_teacher = disp_to_depth(depth_teacher_input.unsqueeze(0), opt.min_depth,
                                                 opt.max_depth)
                depth_teacher = F.interpolate(
                    depth_teacher, [opt.height, opt.width], mode="bilinear", align_corners=False)
                teacher_cam_points = backprojectorT(
                    depth_teacher.squeeze(0), inputs[("inv_K", 0)][jj].unsqueeze(0))
                pix_locs = projectorT(
                    teacher_cam_points, inputs[("K", 0)][jj].unsqueeze(0), T_1[jj].unsqueeze(0))
                if frame_id == 1:
                    warped = F.grid_sample(inputs[("color", 1, 0)].unsqueeze(1)[jj], pix_locs, padding_mode='zeros',
                                           mode='bilinear')
                else:
                    warped = F.grid_sample(inputs[("color", 1, 0)].unsqueeze(1)[jj], pix_locs, padding_mode='zeros',
                                           mode='bilinear')
                diffs_teacher = torch.abs(warped - inputs[("color", 0, 0)].unsqueeze(1)[jj]).mean(1)

                mmin, _ = diffs.min(0)
                cost_mask = mmin.float()
                cost_frame = cost_mask.unsqueeze(0)
                cost_all_frame.append(cost_frame)
                te_cost_all_frame.append(diffs_teacher)

            mask_cur_frame = torch.cat(mask_all_frame, dim=0)
            mask_cur_frame, _ = mask_cur_frame.max(dim=0)
            mask_cur_frame = mask_cur_frame.unsqueeze(0)
            volume_mask.append(mask_cur_frame)

            cost_cur_frame = torch.cat(cost_all_frame, dim=0)
            cost_cur_frame, _ = cost_cur_frame.min(dim=0)
            cost_cur_frame = cost_cur_frame.unsqueeze(0)
            teacher_cur_frame = torch.cat(te_cost_all_frame, dim=0)
            teacher_cur_frame, _ = teacher_cur_frame.min(dim=0)
            teacher_cur_frame = teacher_cur_frame.unsqueeze(0)

            volume_error.append((((cost_cur_frame < teacher_cur_frame) * (teacher_cur_frame > 0.01)) * (
            (cost_cur_frame < 0.002))).float())
        matched_volume_mask = torch.cat(volume_mask, dim=0)
        matched_volume_error = torch.cat(volume_error, dim=0)
        outputs["matched_volume_mask"] = matched_volume_mask
        outputs["matched_photomask"] = matched_volume_error
    return outputs