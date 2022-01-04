import argparse
import os

import torch
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error # 均方误差
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dynamic_channels import sample_tiny_sub_channel

from model import G
from util import is_image, load_image, save_image

parser = argparse.ArgumentParser(description='DeepRendering-implementation')
parser.add_argument('--dataset', required=True, help='unity')
parser.add_argument('--model', type=str, default="checkpoint/cornellbox/netG_model_epoch_0.pth",  help='model file')
parser.add_argument('--accuracy', type=int, default=1,  help='model file')
parser.add_argument('--n_channel_input', type=int, default=3, help='input channel')
parser.add_argument('--n_channel_output', type=int, default=3, help='output channel')
parser.add_argument('--n_generator_filters', type=int, default=64, help="number of generator filters")
opt = parser.parse_args()

assert opt.accuracy < 4 and opt.accuracy > 1

model_list_len=100
answer_list_ss=[]
answer_list_mse=[]
for i in tqdm(range(0,model_list_len)):
    netG_model = torch.load("checkpoint/cornellbox/netG_model_epoch_"+str(i)+".pth")
    netG = G(opt.n_channel_input * 4, opt.n_channel_output, opt.n_generator_filters)
    netG.load_state_dict(netG_model['state_dict_G'])
    root_dir = 'dataset/{}/test/'.format(opt.dataset)
    image_dir = 'dataset/{}/test/albedo'.format(opt.dataset)
    image_filenames = [x for x in os.listdir(image_dir) if is_image(x)]
    cou = 0
    sum_ss = 0
    sum_mse = 0
    for image_name in image_filenames:
        albedo_image = load_image(root_dir + 'albedo/' + image_name)
        direct_image = load_image(root_dir + 'direct/' + image_name)
        normal_image = load_image(root_dir + 'normal/' + image_name)
        depth_image = load_image(root_dir + 'depth/' + image_name)
        gt_image = load_image(root_dir + 'gt/' + image_name)

        albedo = Variable(albedo_image).view(1, -1, 256, 256).cuda()
        direct = Variable(direct_image).view(1, -1, 256, 256).cuda()
        normal = Variable(normal_image).view(1, -1, 256, 256).cuda()
        depth = Variable(depth_image).view(1, -1, 256, 256).cuda()
        sample_tiny_sub_channel(netG, size=opt.accuracy, n_filters=opt.n_generator_filters)
        netG = netG.cuda()
        
        out = netG(torch.cat((albedo, direct, normal, depth), 1))
        out = out.cpu()
        out_img = out.data[0]
        
        out_img_out=out_img.numpy()
        out_img_out=np.transpose(out_img_out,(1,2,0))
        gt_image_out=gt_image.numpy()
        gt_image_out=np.transpose(gt_image_out,(1,2,0))
        #print(out_img.shape,gt_image.shape)
        ss=ssim(out_img_out,gt_image_out,multichannel=True)

        out_img = out_img.add(1).div(2)
        out_img = out_img.numpy()
        out_img *= 255.0
        out_img = out_img.clip(0, 255)
        out_img = np.transpose(out_img, (1, 2, 0))
        out_img = out_img.astype(np.uint8)
        out_img=out_img.reshape(3,-1)

        gt_image = gt_image.add(1).div(2)
        gt_image = gt_image.numpy()
        gt_image *= 255.0
        gt_image = gt_image.clip(0, 255)
        gt_image = np.transpose(gt_image, (1, 2, 0))
        gt_image = gt_image.astype(np.uint8)
        gt_image=gt_image.reshape(3,-1)
        #print(out_img)
        #print(out_img.shape,gt_image.shape)
        mse=mean_squared_error(out_img,gt_image)
        #print(mse)
        sum_ss= sum_ss + ss
        sum_mse = sum_mse + mse
        cou = cou + 1
    print("Epoch {}: ".format(i),"SSIM: ",sum_ss/cou, "MSE: ",sum_mse/cou)
    answer_list_ss.append(sum_ss/cou)
    answer_list_mse.append(sum_mse/cou)
print(answer_list_ss)
print(answer_list_mse)    