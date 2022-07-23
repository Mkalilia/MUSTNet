import torch
from networks.layers import pixel_grid
import torch.nn.functional as F

def flow_mask_generate(inputs, outputs,width, height,batch_size):
    with torch.no_grad():
        flow_forward = inputs[("flow_forward", 0)].squeeze(1)
        flow_backward = inputs[("flow_backward", 0)].squeeze(1)
        flow_forward = F.interpolate(
            flow_forward, [height, width], mode="bilinear", align_corners=False)
        flow_backward = F.interpolate(
            flow_backward, [height, width], mode="bilinear", align_corners=False)
        flow_mask = (torch.sum(torch.abs(flow_forward + flow_backward), dim=1) < 10).detach()
        corrd_3D = outputs[("sample", 1, 0)]
        corrd_3D = corrd_3D / 2 + 0.5
        corrd_3D[..., 0] *= width - 1
        corrd_3D[..., 1] *= height - 1
        corrd_3D = corrd_3D.permute(0, 3, 1, 2)
        grid = pixel_grid((batch_size), (height, width))
        flow_3D = corrd_3D - grid
        inputs[("consistent_mask")] = (flow_mask | (torch.sum(torch.abs(flow_3D - flow_forward), dim=1) < 1)).unsqueeze(
            1).float()
        inputs[("consistent")] = inputs[("consistent_mask")].detach()
        inputs[("flow_3D")] = flow_3D.detach()