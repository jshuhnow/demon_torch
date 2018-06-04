import torch
import torch.nn as nn

from torch.autograd import Variable

from model.subnets import BootstrapNet, IterativeNet, RefinementNet

K = [[0.89115971,  0,  0.5],
     [0,  1.18821287,  0.5],
     [0,           0,    1]]
intrinsics = Variable(torch.Tensor(K), requires_grad=False)



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.bootstrap_net = BootstrapNet()
        self.iterative_net = IterativeNet()
        self.refinement_net = RefinementNet()
    
    def forward(self, img_pair, img1, img2):
        result = self.bootstrap_net(img_pair, img2)

        for i in range(3):
            result = self.iterative_net(img_pair, img2, intrinsics, result)

        result = self.refinement_net(img1, result['depth'])

        return result
