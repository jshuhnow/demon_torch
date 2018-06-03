import numpy as np
import torch

from torch.nn.modules.module import Module
from torch.autograd import Variable, Function


from operators import matrix_inv, axis_angle_to_rotation_matrix, per

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
        return _backward(w, R, grad_out)

    def _backward(selfself, w, R, grad_out):
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

