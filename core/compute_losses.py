from networks.layers import *
from utils.flow_mask import flow_mask_generate

def compute_reprojection_loss(pred, target):
    """
    Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    ssim = SSIM()
    ssim.cuda()
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def weight_loss_metric(opt,outputs):
    """
    Computes weights loss for the 1st traning stage a minibatch
    """
    distance = 0
    for T_idx in range(opt.num_teacher):
        if T_idx+1<opt.num_teacher:
            for idx in range(T_idx+1,opt.num_teacher):
                distance += torch.abs(
                    torch.cosine_similarity(outputs[("weights_{}".format(T_idx))], outputs[("weights_{}".format(idx))],
                                            dim=1))

    mean_distance = torch.mean(distance)
    return mean_distance

def compute_losses_S1(opt, inputs, outputs,epoch):
    """
    Compute the reprojection and smoothness losses for the 1st training stage
    """
    losses = {}
    total_loss = 0
    num_scales = 4

    if opt.num_teacher == None:
        num_d = 1
    else:
        num_d = opt.num_teacher
        if opt.set_assistant_teacher and epoch < opt.use_assistant_epoch:
            num_d = 1

    for D_idx in range(num_d):
        for scale in opt.scales:
            loss = 0
            reprojection_losses = []
            if opt.use_flow:
                flow_mask_generate(inputs, outputs, opt.width, opt.height, opt.batch_size)
            source_scale = 0

            disp = outputs[("disp", 0, scale,D_idx)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            for frame_id in opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale,D_idx)]
                reprojection_losses.append(compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            if not opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses
            if not opt.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if opt.num_teacher != None:
                weight_error = weight_loss_metric(opt, outputs)
                loss += (weight_error * 0.1) / opt.num_teacher

            loss += to_optimise.mean()

            if not opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += opt.disparity_smoothness * (smooth_loss) / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss
    total_loss +=outputs[("norm_constraint")]*0.01
    losses["norm_constraint"] = outputs[("norm_constraint")]

    total_loss /= num_scales
    losses["loss"] = total_loss
    return losses

def compute_pairwise_loss(opt, outputs, losses):
    """
    Computes geometric loss for a minibatch
    """
    total_loss = 0
    num_scales = 4
    for scale in opt.scales:
        diff_depths = []
        computed_depth = outputs[("depth", 0, scale)]
        for f_i in opt.frame_ids[1:]:
            projected_depth = outputs[("depth", f_i, scale)]

            diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
            diff_depths.append(diff_depth)
        error_min, _ = torch.min(torch.cat(diff_depths, dim=1), dim=1)
        losses[("geometric_error", scale)] = (error_min.unsqueeze(1) * outputs[("valid_mask", scale)]).sum() / (
            outputs[("valid_mask", scale)].sum())
        total_loss += losses[("geometric_error", scale)]
    losses["loss"] += total_loss / num_scales * 0.00001
    return losses

def compute_depth_losses(opt, inputs, outputs, losses):
    disp_pred = outputs[("disp", 0, 0)]
    _, depth_pred = disp_to_depth(disp_pred, opt.opt.min_depth, opt.opt.max_depth)
    depth_pred = torch.clamp(F.interpolate(
        depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_pred = depth_pred.detach()

    depth_gt = inputs["depth_gt"]
    mask = depth_gt > 0

    # garg/eigen crop
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)

    for i, metric in enumerate(opt.depth_metric_names):
        losses[metric] = np.array(depth_errors[i].cpu())

