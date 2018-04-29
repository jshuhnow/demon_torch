import torch.nn as nn
from operation import WarpImageLayer, DepthToFlowLayer, FlowToDepthLawer

class FlowBlock(nn.Module):

    def __init__(self, is_bootstrap=False):
        super(FlowBlock, self).__init__()

    def forward(self, img_pair, img2_2=None, intrinsics=None):
        pass

class DepthMotionBlock(nn.Module):

    def __init__(self, is_bootstrap=False):
        super(DepthMotionBlock, self).__init__()

    def forward(self):
        pass

class RefinementBlock(nn.Module):

    def __init__(self):
        super(RefinementBlock, self).__init__()

    def forward(self):
        pass
