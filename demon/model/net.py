import torch
import torch.nn as nn

from torch.autograd import Variable

from .subnets import BootstrapNet, IterativeNet, RefinementNet




class Net(nn.Module):

    def __init__(self, K):
        
        super(Net, self).__init__()
        self.intrinsics = Variable(torch.Tensor(K), requires_grad=False)
        self.bootstrap_net = BootstrapNet()
        self.iterative_net = IterativeNet()
        self.refinement_net = RefinementNet()
        
    
    def forward(self, img_pair, img1, img2):
        result = self.bootstrap_net(img_pair, img2)

        for i in range(3):
            result = self.iterative_net(img_pair, img2, self.intrinsics, result)

        result = self.refinement_net(img1, result['depth'])

        return result
