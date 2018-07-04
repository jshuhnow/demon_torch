import torch.nn as nn

class WarpImgLayer(nn.Module):
    
    def __init__(self):
        super(WarpImgLayer, self).__init__()
        
    def forward(self):
        pass

class DepthToFlowLayer(nn.Module):
    
    def __init__(self):
        super(DepthToFlowLayer, self).__init__()
        
    def forward(self):
        pass
    
class FlowToDepthLayer(nn.Module):
    
    def __init__(self, normalized_K):
        super(FlowToDepthLayer, self).__init__()
        self.normalized_K = normalized_K
        
    def forward(self):
        pass
