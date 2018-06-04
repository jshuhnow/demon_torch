import torch
import torch.nn as nn
import lmbspecialops as sops

def conv_leakyrelu2_block(input_channels, output_channels, kernel_size, stride, leaky_coeff=0.1):
    """
    For computational efficency, split the 2D convolution to two 1D convolution.
    It follows the Caffe style.

    :param input_channels: # of input channels 
    :param output_channels: # of output channels
    :param kernel_size: kernel size
    :param stride: stride
    :param leaky_coeff: leaky ReLU's coefficient
    :return: sequential layer of (conv+leakyReLU)x2
    """

    # stride
    s = None
    if not isinstance(stride, tuple):
        s = (stride, stride)
    else:
        s = stride
        
    conv_1 = nn.Conv2d(input_channels, output_channels,
                        (kernel_size[0], 1),
                        padding=(kernel_size[0] // 2, 0),
                        stride=(s[0], 1))
    leaky_relu_1 = nn.LeakyReLU(leaky_coeff)

    conv_2 = nn.Conv2d(output_channels, output_channels,
                        (1,kernel_size[1]),
                        padding=(0, kernel_size[1] // 2),
                        stride=(1, s[1]))
    leaky_relu_2 = nn.LeakyReLU(leaky_coeff)
    
    return nn.Sequential(
        conv_1,
        leaky_relu_1,
        conv_2,
        leaky_relu_2
    )

def conv_leakyrelu_block(input_channels, output_channels, kernel_size, stride, leaky_coeff=0.1):
    """
    It follows the Caffe style.

    :param input_channels:
    :param output_channels:
    :param kernel_size:
    :param stride:
    :param leaky_coeff:
    :return: sequential layer of conv + leakyReLU
    """

    if not isinstance(stride, tuple):
        s = (stride, stride)
    else:
        s = stride

    conv = nn.Conv2d(input_channels, output_channels,
                     kernel_size,
                     padding=(kernel_size[0] // 2, kernel_size[1] //2),
                     stride=s)
    leaky_relu = nn.LeakyReLU(leaky_coeff)

    return nn.Sequential(
        conv,
        leaky_relu
    )

def upconv_leakyrelu_block(input_channels, output_channels=4, leaky_coeff =0.1):
    kernel_size = (4,4)
    stride = (2,2)
    padding = 1

    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels,
                           kernel_size, stride, padding),
        nn.LeakyReLU(leaky_coeff)
    )

def _predict_flow_block(input_channels, output_channels=4, intermediate_channels=24):

    conv1 = conv_leakyrelu_block(input_channels, intermediate_channels, (3,3), 1)
    conv2 = nn.Conv2d(intermediate_channels, output_channels, (3,3), padding=(1,1), stride=1)

    return nn.Sequential(
        conv1,
        conv2
    )

def _predict_motion_block(num_inputs, leaky_coeff=0.1):
    """

    :param num_inputs:
    :param leaky_coeff:
    :return:
    """

    conv1 = conv_leakyrelu_block(num_inputs, 128, (3, 3), 1)
    fc1 = nn.Linear(128 * 8 * 6, 1024)
    leaky_relu1 = nn.LeakyReLU(leaky_coeff)
    fc2 = nn.Linear(1024, 128)
    leaky_relu2 = nn.LeakyReLU(leaky_coeff)
    fc3 = nn.Linear(128, 7)

    return conv1, nn.Sequential(
        fc1, leaky_relu1, fc2, leaky_relu2, fc3
    )

class FlowBlock(nn.Module):

    def __init__(self, given_predictions = False):
        super(FlowBlock, self).__init__()
        self.conv1 = conv_leakyrelu2_block(3 * 2, 32, (9, 9), 2)


        self.conv2 = conv_leakyrelu2_block(32, 64, (7,7), 2)
        self.conv2_1 = conv_leakyrelu2_block(64, 64, (3,3), 1)

        if given_predictions:
            self.warp_image = WarpImgLayer()
            self.depth_to_flow = DepthToFlowLayer()
            self.conv2_extra_inputs = conv_leakyrelu2_block(9, 32, (3, 3), 1)

        self.conv3 = conv_leakyrelu2_block(64, 128, (5, 5), 2)
        self.conv3_1 = conv_leakyrelu2_block(128, 128, (3,3), 1)

        self.conv4 = conv_leakyrelu2_block(128, 256, (5,5), 2)
        self.conv4_1 = conv_leakyrelu2_block(256, 256, (3, 3), 1)

        self.conv5 = conv_leakyrelu2_block(256, 512, (5,5), 2)
        self.conv5_1 = conv_leakyrelu2_block(512, 512, (3,3), 1)

        self.flow5 = _predict_flow_block(512, 4)
        self.flow5_upconv = nn.ConvTranspose2d(4, 2, (4,4), stride=(2,2), padding=1)

        self.upconv5 = upconv_leakyrelu_block(512, 256)
        self.upconv4 = upconv_leakyrelu_block(516, 128)
        self.upconv3 = upconv_leakyrelu_block(256, 64)

        self.flow2 = _predict_flow_block(128, 4)


    def forward(self, img_pair, img2_2=None, intrinsics=None, prediction=None):
        """

        :param img_pair:
        :param img2_2:
        :param intrinsics:
        :param given_prediction:
        :return:
        """

        conv1 = self.conv1(img_pair)
        conv2 = self.conv2(conv1)

        conv2_1 = None
        if not prediction:
            conv2_1 = self.conv2_1(conv2)
        else:
            depth = self.prediction['depth']
            normal = self.prediction['normal']
            r = self.prediction['r']
            t = self.prediction['t']

            flow = self.depth_to_flow(intrinsics, intrinsics, depth, r, t)

            warpped_img = self.warp_image(img2_2, flow)
            concat_prediction = torch.cat((warpped_img, flow, depth, normal), 1)
            extra2 = self.conv2_extra_inputs(concat_prediction)
            conv2_1 = self.conv2_1(torch.cat((conv2, extra2), 1))

        conv3 = self.conv3(conv2_1)
        conv3_1 = self.conv3_1(conv3)
        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)
        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)

        upconv5 = self.upconv5(conv5_1)
        flow5 = self.flow5(conv5_1)
        flow5_upconv = self.upconv(flow5)

        upconv4 = self.upconv4(torch.cat( (upconv5, conv4_1, flow5_upconv), 1))
        upconv3 = self.upconv3(torch.cat( (upconv4, conv3_1), 1))
        flow2 = self.flow2(torch.cat( (upconv3, conv2_1), 1))

        return flow2


class DepthMotionBlock(nn.Module):

    def __init__(self, given_motion=False):
        super(DepthMotionBlock, self).__init__()

        self.conv1 = conv_leakyrelu2_block(6, 32, (9,9), 2)
        self.conv2 = conv_leakyrelu2_block(32, 32, (7, 7), 2)

        self.warpped_img = WarpImgLayer()

        self.conv2_extra = None
        if given_motion:
            self.conv2_extra = conv_leakyrelu2_block(8, 32, (3,3), 1)
            self.flow_to_depth = FlowToDepthLayer(normalized_K = True)
        else:
            self.conv2_extra = conv_leakyrelu2_block(7, 32, (3,3), 1)

        self.conv2_1 = conv_leakyrelu2_block(64, 64, (3,3), 1)
        self.conv3 = conv_leakyrelu2_block(64, 128, (5,5), 2)
        self.conv3_1 = conv_leakyrelu2_block(128, 128, (3,3), 1)
        self.conv4 = conv_leakyrelu2_block(128, 256, (5,5), 2)
        self.conv4_1= conv_leakyrelu2_block(256, 256, (3,3), 1)
        self.conv5 = conv_leakyrelu2_block(256, 512, (3,3), 2)
        self.conv5_1 = conv_leakyrelu2_block(512, 512, (3,3), 1)

        self.upconv4 = upconv_leakyrelu_block(512, 256)
        self.upconv3 = upconv_leakyrelu_block(514, 128)
        self.upconv2 = upconv_leakyrelu_block(256, 64)
        self.flow2 = predict_flow_block(128, num_outputs=4)

    def forward(self, img_pair, img2_2, prv_flow2, prv_flowconf2, prediction=None, intrinsics=None):
        """

        :param img_pair:
        :param img2_2:
        :param prv_flow2:
        :param prv_flowconf2:
        :param prediction:
        :param intrinsics:
        :return:
        """

        conv1 = self.conv1(img_pair)
        conv2 = self.conv2(conv1)

        warpped_img = self.warpped_img(img2_2, prv_flow2)

        concat2 = None
        if not prediction:
            concat2 = torch.cat( (warpped_img, prv_flowconf2), 1)
        else:
            r, t= prediction['r'], prediction['t']
            depth = self.flow_to_depth(intrinsics, intrinsics, prv_flow2, r, t)

            concat2 = torch.cat ((warpped_img, prv_flowconf2, depth), 1)

        extra2 = self.conv2_extra(concat2)
        conv2_1 = self.conv2_1(torch.cat((conv2, extra2), 1))
        conv3 = self.conv3(conv2_1)
        conv3_1 = self.conv3_1(conv3)
        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4(conv4)
        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)

        upconv4 = self.upconv4(conv5_1)
        upconv3 = self.upconv3( torch.cat((upconv4, conv4_1), 1))
        upconv2 = self.upconv2( torch.cat((upconv3, conv3_1), 1))

        depth, normal = self.depth_normal( torch.cat((upconv3, conv2_1), 1))
        motion_conv = self.motion_conv(conv5_1)

        motion = self.motion_fc(
            motion_conv.view(
                motion_conv.size(0),
                128 * 6 * 8
            )
        )

        r = motion[:, 0:3]
        t = motion[:, 3:6]
        scale = motion[:, 6]

        return {
            'depth' : depth,
            'normal' : normal,
            'r' : r,
            't' : t
        }



class RefinementBlock(nn.Module):

    def __init__(self):
        super(RefinementBlock, self).__init__()
        self.conv0 = conv_leakyrelu_block(4, 32, (3,3), (1,1))
        self.conv1 = conv_leakyrelu_block(32, 64, (3,3), (2,2))
        self.conv1_1= conv_leakyrelu_block(64,64,(3,3), (1,1))
        self.conv2 = conv_leakyrelu_block(64, 128, (3,3), (2,2))
        self.covn2_1 = conv_leakyrelu_block(128, 128,(3,3),(1,1))
        self.upconv2 = upconv_leakyrelu_block(128, 64)
        self.upconv1 = upconv_leakyrelu_block(128, 32)
        self.depth_refine = predict_flow_block(64, num_outputs = 1, intermediate_num_outputs=16)

    def forward(self, img1, depth):
        """

        :param img1:
        :param depth:
        :return:
        """

        W, H = img1.shape[-2:]
        upsampler = nn.Upsample(size=(H,W), mode='nearest')

        depth_upsampled = upsampler(depth)
        concat0 = torch.cat(
            torch.autograd.Variable(
                torch.from_numpy(img1),
                requires_grad = False),
            depth_upsampled
            ), 1


        conv0 = self.conv0(concat0)
        conv1 = self.conv1(conv0)
        conv1_1 = self.conv1_1(conv1)
        conv2 = self.conv2(conv1_1)
        conv2_1 = self.conv2_1(conv2)

        upconv2 = self.upconv2(conv2_1)
        upconv1 = self.upconv1(torch.cat(upconv2, conv1_1), 1)

        refined_depth = self.depth_refine(torch.cadt((upconv1, conv0), 1))
        return refined_depth



