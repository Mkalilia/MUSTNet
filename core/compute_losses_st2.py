import torch
from networks.layers import *
from utils.utils import Sobel
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

def compute_losses_S2(opt, inputs, outputs):
    """
    Compute the distilation loss for the 2nd training stage
    """
    losses = {}
    total_loss = 0
    total_loss_depth = 0
    num_scales = 4

    target = inputs[("color", 0, 0)]
    teacher_repro = []
    for frame_id in opt.frame_ids[1:]:
        pred_teacher = outputs[("color_teacher", frame_id, 0)]
        teacher_repro.append(compute_reprojection_loss(pred_teacher, target))
    teacher_repros, _ = torch.min(torch.cat(teacher_repro, dim=1), dim=1)

    for scale in opt.scales:
        loss = 0
        reprojection_losses = []
        if opt.use_flow:
            flow_mask_generate(inputs, outputs, opt.width, opt.height, opt.batch_size)
        source_scale = 0

        disp = outputs[("disp", 0, scale, 0)]
        color = inputs[("color", 0, scale)]
        target = inputs[("color", 0, source_scale)]

        for frame_id in opt.frame_ids[1:]:
            pred = outputs[("color", frame_id, scale, 0)]
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

        # the select mask for photometric reprojection loss
        if opt.use_flow:
            photo_mask = inputs[("consistent_mask")] * outputs["matched_volume_mask"] * outputs["matched_photomask"]
        else:
            photo_mask = outputs["matched_volume_mask"] * outputs["matched_photomask"]

        if opt.use_flow:
            to_optimise_mean = (inputs[("consistent_mask")] * to_optimise * photo_mask).sum() / (
                (inputs[("consistent_mask")] * outputs["matched_volume_mask"] * outputs[
                    "matched_photomask"]).sum()).detach()
        else:
            to_optimise_mean = (to_optimise * photo_mask).sum() / (
                (outputs["matched_volume_mask"] * outputs[
                    "matched_photomask"]).sum()).detach()
        loss += to_optimise_mean

        # depth distillation loss
        target_depth = inputs[("depth_p", 0)]
        student_disp = outputs[("disp", 0, scale, 0)]

        cos = nn.CosineSimilarity(dim=1, eps=0)
        get_gradient = Sobel().cuda()
        target_dpt = target_depth.unsqueeze(1).detach()

        target_dpt = F.interpolate(
            target_dpt, [opt.height, opt.width],
            mode="bilinear", align_corners=False)
        pre_dpt_out = F.interpolate(
            student_disp, [opt.height, opt.width],
            mode="bilinear", align_corners=False)

        depth_grad = get_gradient(target_dpt)
        output_grad = get_gradient(pre_dpt_out)

        ones = torch.ones(pre_dpt_out.size(0), 1, pre_dpt_out.size(2),
                          pre_dpt_out.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(target_dpt)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(target_dpt)
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(target_dpt)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(target_dpt)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        gt_height, gt_width = target_dpt.shape[2:]
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = torch.tensor(np.zeros(target_dpt.shape)).cuda()
        crop_mask[:, :, crop[0]:crop[1], crop[2]:crop[3]] = 1

        # the select mask for depth distillation
        if opt.use_flow:
            depth_mask = inputs[("consistent_mask")] * (
                (outputs["matched_volume_mask"] * (1 - outputs["matched_photomask"])).unsqueeze(1))
        else:
            depth_mask = (
                (outputs["matched_volume_mask"] * (1 - outputs["matched_photomask"])).unsqueeze(1))

        # loss depth
        loss_depth = (
                (((
                      torch.abs(pre_dpt_out - target_dpt)) * depth_mask).sum(-1).sum(-1).sum(-1)) / (
                    depth_mask).sum(-1).sum(-1).sum(
            -1)).mean()
        loss_dx = (((
                        torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 1)) * depth_mask).sum(
            -1).sum(-1).sum(-1) / depth_mask.sum(-1).sum(
            -1).sum(-1)).mean()
        loss_dy = ((((
                         torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 1)) * depth_mask).sum(
            -1).sum(-1).sum(-1)) / depth_mask.sum(-1).sum(
            -1).sum(-1)).mean()
        loss_normal = (((
                                torch.abs(1 - cos(output_normal, depth_normal)) * depth_mask).sum(-1).sum(
            -1).sum(-1)) / depth_mask.sum(-1).sum(-1).sum(
            -1)).mean()

        loss += ((loss_depth) + 0.2 * loss_normal + 0.2 * (loss_dx + loss_dy))
        total_loss_depth += ((loss_depth) + 0.2 * loss_normal + 0.2 * (loss_dx + loss_dy))
        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)
        loss += opt.disparity_smoothness * (smooth_loss) / (2 ** scale)

        total_loss += loss
        losses["loss/{}".format(scale)] = loss

    total_loss_depth /= num_scales
    losses["loss_depth"] = total_loss_depth
    losses["loss"] = total_loss
    return losses































