import torch.nn as nn
from model.blocks import *

leaky_coeff = 0.1


"""
BootstrapNet
"""
class BootstrapNet(nn.Module):

    def __init__(self):
        super(BootstrapNet, self).__init__()
        self.flow_block = FlowBlock(given_predictions=False)
        self.depth_motion_block = DepthMotionBlock(given_motion=False)

    def forward(self, img_pair, img2_2):
        flow = self.flow_block(img_pair, img2_2)
        prediction = self.depth_motion_block(img_pair, img2_2, flow[:,:2,:,:], flow)

        return prediction

"""
IterativeNet
"""
class IterativeNet(nn.Module):
    def __init__(self):
        super(IterativeNet, self).__init__()

        self.flow_block = FlowBlock(given_predictions=True)
        self.depth_motion_block = DepthMotionBlock(given_motion=True)

    def forward(self, img_pair, img2_2, intrinsics, prv_prediction):
        flow = self.flow_block(img_pair, img2_2, intrinsics, prv_prediction)
        prediction = self.depth_motion_block(img_pair,img2_2,flow[:, :2, :, :], flow,
                                             prediction=prv_prediction, intrinsics=intrinsics)
        return prediction

"""
RefinementNet
"""
class RefinementNet(nn.Module):
    def __init__(self):
        super(RefinementNet, self).__init__()

        self.refinement_block = RefinementBlock()

    def forward(self, img1, depth):
        refinement= self.refinement_block(img1, depth)
        return refinement


