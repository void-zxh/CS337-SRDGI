import argparse
import os
import time

import torch
from torch.autograd import Variable
from dynamic_channels import sample_tiny_sub_channel
from model import G
from util import is_image, load_image, save_image

parser = argparse.ArgumentParser(description='DeepRendering-implementation')
parser.add_argument('--dataset', required=True, help='unity')
parser.add_argument('--model', type=str, required=True, help='model file')
parser.add_argument('--accuracy', type=int, default=1,  help='model file')
parser.add_argument('--n_channel_input', type=int, default=3, help='input channel')
parser.add_argument('--n_channel_output', type=int, default=3, help='output channel')
parser.add_argument('--n_generator_filters', type=int, default=64, help="number of generator filters")
opt = parser.parse_args()

netG_model = torch.load(opt.model)
netG = G(opt.n_channel_input * 4, opt.n_channel_output, opt.n_generator_filters)
netG.load_state_dict(netG_model['state_dict_G'])
root_dir = 'dataset/{}/test/'.format(opt.dataset)
image_dir = 'dataset/{}/test/albedo'.format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image(x)]
time_list=[]

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
    
    start_p=time.time()
    out = netG(torch.cat((albedo, direct, normal, depth), 1))
    end_p=time.time()
    out = out.cpu()
    out_img = out.data[0]
    time_list.append(end_p-start_p)

    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(os.path.join("result", "accuracy_{}".format(opt.accuracy))):
        os.mkdir(os.path.join("result", "accuracy_{}".format(opt.accuracy)))
    save_image(out_img, "result/accuracy_{}/{}".format(opt.accuracy, image_name))
    save_image(gt_image, "result/accuracy_{}/GT{}".format(opt.accuracy, image_name))
print(time_list)