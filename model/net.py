import torch.nn as nn

from subnets import BootstrapNet, IterativeNet, RefinementNet

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init()

        self.bootstrap_net = BootstrapNet()
        self.iterative_net = IterativeNet()
        self.refinement_net = RefinementNet()
    
    def __forward__(self):
        pass
