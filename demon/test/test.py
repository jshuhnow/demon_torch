from PIL import Image
from torch.autograd import Variable
import os
import numpy as np
import torch
from model.net import Net

def prepare_input_data(img1, img2, data_format):
    """Creates the arrays used as input from the two images."""
    # scale images if necessary
    if img1.size[0] != 256 or img1.size[1] != 192:
        img1 = img1.resize((256,192))
    if img2.size[0] != 256 or img2.size[1] != 192:
        img2 = img2.resize((256,192))
    img2_2 = img2.resize((64,48))
        
    # transform range from [0,255] to [-0.5,0.5]
    img1_arr = np.array(img1).astype(np.float32)/255 -0.5
    img2_arr = np.array(img2).astype(np.float32)/255 -0.5
    img2_2_arr = np.array(img2_2).astype(np.float32)/255 -0.5
    
    if data_format == 'channels_first':
        img1_arr = img1_arr.transpose([2,0,1])
        img2_arr = img2_arr.transpose([2,0,1])
        img2_2_arr = img2_2_arr.transpose([2,0,1])
        image_pair = np.concatenate((img1_arr,img2_arr), axis=0)
    else:
        image_pair = np.concatenate((img1_arr,img2_arr),axis=-1)
    
    result = {
        'image_pair': image_pair[np.newaxis,:],
        'image1': img1_arr[np.newaxis,:], # first image
        'image2_2': img2_2_arr[np.newaxis,:], # second image with (w=64,h=48)
    }
    return result




examples_dir='.'
img1 = Image.open(os.path.join(examples_dir, 'sculpture1.png'))
img2 = Image.open(os.path.join(examples_dir, 'sculpture2.png'))

input_data = prepare_input_data(img1, img2, 'channels_first')

img_pair = Variable( torch.FloatTensor(input_data['image_pair']), requires_grad=False )
img1 =Variable( torch.FloatTensor(input_data['image1']), requires_grad=False )
img2_2= Variable( torch.FloatTensor(input_data['image2_2']), requires_grad=False )

net = Net()
result = net(img_pair, img1, img2_2)
print(result)

