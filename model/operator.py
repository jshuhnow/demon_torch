import numpy as np
import torch

from torch.nn.modules.module import Module
from torch.autograd import Variable, Function


import lmbspecialops as sops

class Exp(Function):
    def forward(self, w):
        R = self._forward(w)
        self.save_for_backward(w, R)
        return R

    @staticmethod
    def _forward(self,w):
        theta = torch.norm(w)

        if theta > 0:
            wx = torch.Tensor([
                [0, -w[2], w[1]],
                [w[2], 0, -w[0]],
                [-w[1], w[0], 0]
            ])
            R = torch.eye(3) + np.sin(theta) / theta *wx + ((1-np.cos(theta)) / theta ** 2) * wx.mm(wx)
        else:
            R = torch.Tensor([
                [1,-w[2], w[1]],
                [w[2], 1, -w[0]],
                [-w[1], w[0], 1]
            ])
        return R

    def backward(self, grad_out):
        w, R = self.saved_tensors
        return self._backward(w, R, grad_out)

    def _backward(self,w ,R, grad_out):
        grad_w = torch.zeros(3)
        theta = torch.norm(w)

        if theta > 0:
            wx = torch.Tensor([
                [0, -w[2], w[1]],
                [w[2], 0, -w[0]],
                [-w[1], w[0], 0]
            ])

            for i in range(3):
                ei = torch.zeros(3, 1)
                ei[i] = 1

def exp(w):
    return Exp(w)


class WarpImgLayer(Module):
    def __init__(self):
        super(WarpImgLayer, self).__init__()

    def forward(self, img, flow, normalized= True, border_mode = 'value'):
        return sops.warp2d(img, flow, normalized=normalized, border_mode=border_mode)



class DepthToFlowLayer(Module):
    def __init__(self):
        super(DepthToFlowLayer, self).__init__()

    def forward(self, intrinsics, depth, rotation, translation, inverse_depth, normalized_flow):
        return  sops.depth_to_flow(
                intrinsics = intrinsics,
                depth = depth,
                rotation = rotation,
                translation = translation,
                inverse_depth=inverse_depth,
                normalized_flow = normalized_flow)

class FlowToDepthLayer(Module):
    def __init__(self):
        super(FlowToDepthLayer, self).__init__()

    def forward(self,  flow, intrinsics, rotation, translation, normalized_flow, inverse_depth):
        return sops.flow_to_depth(
                flow=flow,
                intrinsics=intrinsics,
                rotation=rotation,
                translation=translation,
                normalized_flow =normalized_flow,
                inverse_depth=inverse_depth)
