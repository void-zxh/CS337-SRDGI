import random
import math
import torch
from ops import *
from model import G
CHANNEL_CONFIGS = [0.25, 0.5, 0.75, 1.0]
G_CHANNEL_CONFIG = [1, 2, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 2, 1, 0, 1, 2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8]

def get_full_channel_configs(model):# get the output channels
    full_channels = []
    for m in model.modules():
        if isinstance(m, myConvTranspose2d):
            full_channels.append(m.weight.shape[0])
        elif isinstance(m, EqualConv2d):
            full_channels.append(m.weight.shape[0])  
        elif isinstance(m, myBatchNorm2d):
            full_channels.append(m.gamma.shape[1])
    return full_channels


def set_sub_channel_config(model, sub_channels):
    ptr = 0
    for m in model.modules():
        if isinstance(m, EqualConv2d):
            m.first_k_oup = max(sub_channels[ptr],3)
            ptr += 1
        elif isinstance(m, myBatchNorm2d):
            m.first_k_oup = sub_channels[ptr]
            ptr += 1
        elif isinstance(m,myConvTranspose2d):
            m.first_k_oup = max(sub_channels[ptr],3)
            ptr += 1
    assert ptr == len(sub_channels), (ptr, len(sub_channels))  # all used


def set_uniform_channel_ratio(model, ratio, n_filters):
    full_channels = get_full_channel_configs(model)

    channel_config = [min(v* n_filters, int(v * ratio* n_filters)) for v in G_CHANNEL_CONFIG]

    set_sub_channel_config(model, channel_config)


def remove_sub_channel_config(model):
    for m in model.modules():
        if hasattr(m, 'first_k_oup'):
            del m.first_k_oup


def reset_generator(model):
    remove_sub_channel_config(model)
    if hasattr(model, 'target_res'):
        del model.target_res


def get_current_channel_config(model):
    ch = []
    for m in model.modules():
        if hasattr(m, 'first_k_oup'):
            ch.append(m.first_k_oup)
    return ch


def _get_offical_sub_channel_config(ratio, org_channel_mult = 1):
    channel_max = 128
    # NOTE: 
    # in Python 3.6 onwards, the order of dictionary insertion is preserved
    channel_config = [min(channel_max, int(v * ratio * org_channel_mult)) for _, v in G_CHANNEL_CONFIG.items()]
    return channel_config


def get_random_channel_config(full_channels, org_channel_mult=1, min_channel=3, divided_by=1):
    # use the official config as the smallest number here (so that we can better compare the computation)
    bottom_line = _get_offical_sub_channel_config(CHANNEL_CONFIGS[0])
    bottom_line = bottom_line[:len(full_channels)]

    new_channels = []
    ratios = []
    for full_c, bottom in zip(full_channels, bottom_line):
        valid_channel_configs = [a for a in CHANNEL_CONFIGS if a * full_c >= bottom]  # if too small, discard the ratio
        ratio = random.choice(valid_channel_configs)
        ratios.append(ratio)
        c = int(ratio * full_c)
        c = min(max(c, min_channel), full_c)
        c = math.ceil(c * 1. / divided_by) * divided_by
        new_channels.append(c)
    return new_channels, ratios

def sample_tiny_sub_channel(model, min_channel=8, divided_by=1, size = 1, n_filters=64, mode='uniform', set_channels=True):
    if mode == 'uniform':
        if set_channels:
            set_uniform_channel_ratio(model, CHANNEL_CONFIGS[size-1], n_filters= n_filters)
            print("channel config:", CHANNEL_CONFIGS[size-1])
        return [CHANNEL_CONFIGS[size-1]] * len(get_full_channel_configs(model))
    else:
        raise NotImplementedError

def sample_random_sub_channel(model, min_channel=3, n_filters=64, divided_by=1, seed=None, mode='uniform', set_channels=True):
    if seed is not None:  # whether to sync between workers
        random.seed(seed)
    if mode == 'uniform':
        rand_ratio = random.choice(CHANNEL_CONFIGS)
        # print(rand_ratio)
        if set_channels:
            set_uniform_channel_ratio(model, rand_ratio, n_filters=n_filters)
        return [rand_ratio] * len(get_full_channel_configs(model))
    elif mode == 'flexible':# this mode is not recommend
        full_channels = get_full_channel_configs(model)
        rand_channels, rand_ratios = get_random_channel_config(full_channels,1, min_channel, divided_by)
        if set_channels:
            set_sub_channel_config(model, rand_channels)
        return rand_ratios
    else:
        raise NotImplementedError

#NOTE:
#We have implemented code for dynamic channel sorting.
#However, it does not work well and is therefore not recommended.
def sort_channel(g):
    def _get_sorted_input_idx(conv, cated= True):
        if isinstance(conv, EqualConv2d):
            importance = torch.sum(torch.abs(conv.weight.data), dim=(0, 2, 3))
            return torch.sort(importance, dim=0, descending=True)[1]
        elif isinstance(conv, myConvTranspose2d):
            if cated:
                weight = conv.weight.data[0:int(conv.weight.data.shape[0]/2)]
                importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
            else: 
                importance = torch.sum(torch.abs(conv.weight.data), dim=(1, 2, 3))
            return torch.sort(importance, dim=0, descending=True)[1]

    def _reorg_input_channel(conv, idx, cated= True):
        # print(idx.numel(),conv.weight.data.shape[1])
        if isinstance(conv,myConvTranspose2d) and cated:
            # print(idx)
            rhalf = torch.tensor(range(int(conv.weight.data.shape[0]/2),int(conv.weight.data.shape[0])))
            rhalf = rhalf.cuda()
            # print(rhalf)
            idx = torch.cat((idx,rhalf))
            # print(idx,conv.weight.data.shape[0], idx.numel())
            assert idx.numel() ==conv.weight.data.shape[0]
            conv.weight.data = torch.index_select(conv.weight.data, 0, idx)
        elif isinstance(conv, EqualConv2d):
            assert idx.numel() ==conv.weight.data.shape[1]
            conv.weight.data = torch.index_select(conv.weight.data, 1, idx)  # inp
        elif isinstance(conv,myConvTranspose2d):
            assert idx.numel() ==conv.weight.data.shape[0]
            conv.weight.data = torch.index_select(conv.weight.data, 0, idx)


    def _reorg_output_channel(conv, idx):
        
        if isinstance(conv, EqualConv2d):
            assert idx.numel() == conv.weight.data.shape[0]
            conv.weight.data = torch.index_select(conv.weight.data, 0, idx)  # oup
            conv.bias.data = conv.bias.data[idx]
        elif isinstance(conv, myBatchNorm2d):
            assert idx.numel() == conv.gamma.data.shape[1]
            conv.gamma.data = torch.index_select(conv.gamma.data, 1, idx)
            conv.beta.data = torch.index_select(conv.beta.data, 1, idx)
        elif isinstance(conv, myConvTranspose2d):
            assert idx.numel() == conv.weight.data.shape[1]
            conv.weight.data = torch.index_select(conv.weight.data, 1, idx)  # oup
            conv.bias.data = conv.bias.data[idx]

    sorted_idx = None 
    sorted_idx = _get_sorted_input_idx(g.deconv8)
    _reorg_input_channel(g.deconv8, sorted_idx)

    _reorg_output_channel(g.deconv7, sorted_idx)
    _reorg_output_channel(g.batch_norm, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.deconv7)
    _reorg_input_channel(g.deconv7, sorted_idx)

    _reorg_output_channel(g.deconv6, sorted_idx)
    _reorg_output_channel(g.batch_norm2_2, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.deconv6)
    _reorg_input_channel(g.deconv6, sorted_idx)

    _reorg_output_channel(g.deconv5, sorted_idx)
    _reorg_output_channel(g.batch_norm4_2, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.deconv5)
    _reorg_input_channel(g.deconv5, sorted_idx)

    _reorg_output_channel(g.deconv4, sorted_idx)
    _reorg_output_channel(g.batch_norm8_8, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.deconv4)
    _reorg_input_channel(g.deconv4, sorted_idx)

    _reorg_output_channel(g.deconv3, sorted_idx)
    _reorg_output_channel(g.batch_norm8_7, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.deconv3)
    _reorg_input_channel(g.deconv3, sorted_idx)

    _reorg_output_channel(g.deconv2, sorted_idx)
    _reorg_output_channel(g.batch_norm8_6, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.deconv2)
    _reorg_input_channel(g.deconv2, sorted_idx)

    _reorg_output_channel(g.deconv1, sorted_idx)
    _reorg_output_channel(g.batch_norm8_5, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.deconv1,cated=False)
    _reorg_input_channel(g.deconv1, sorted_idx, cated=False)

    _reorg_output_channel(g.conv8, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.conv8)
    _reorg_input_channel(g.conv8, sorted_idx)

    _reorg_output_channel(g.conv7, sorted_idx)
    _reorg_output_channel(g.batch_norm8_4, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.conv7)
    _reorg_input_channel(g.conv7, sorted_idx)

    _reorg_output_channel(g.conv6, sorted_idx)
    _reorg_output_channel(g.batch_norm8_3, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.conv6)
    _reorg_input_channel(g.conv6, sorted_idx)

    _reorg_output_channel(g.conv5, sorted_idx)
    _reorg_output_channel(g.batch_norm8_2, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.conv5)
    _reorg_input_channel(g.conv5, sorted_idx)

    _reorg_output_channel(g.conv4, sorted_idx)
    _reorg_output_channel(g.batch_norm8_1, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.conv4)
    _reorg_input_channel(g.conv4, sorted_idx)

    _reorg_output_channel(g.conv3, sorted_idx)
    _reorg_output_channel(g.batch_norm4_1, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.conv3)
    _reorg_input_channel(g.conv3, sorted_idx)

    _reorg_output_channel(g.conv2, sorted_idx)
    _reorg_output_channel(g.batch_norm2_1, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.conv2)
    _reorg_input_channel(g.conv2, sorted_idx)

    _reorg_output_channel(g.conv1, sorted_idx)


