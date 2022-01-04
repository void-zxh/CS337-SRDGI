import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.models as models
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(n_filters * 8, n_filters * 8, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(n_filters * 8, n_filters * 8, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 8, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(n_filters * 8 * 2, n_filters * 4, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(n_filters * 4 * 2, n_filters * 2, 4, 2, 1)
        self.deconv7 = nn.ConvTranspose2d(n_filters * 2 * 2, n_filters, 4, 2, 1)
        self.deconv8 = nn.ConvTranspose2d(n_filters * 2, n_channel_output, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(n_filters)
        self.batch_norm2_1 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm2_2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4_1 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm4_2 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8_1 = nn.BatchNorm2d(n_filters * 8)
        self.batch_norm8_2 = nn.BatchNorm2d(n_filters * 8)
        self.batch_norm8_3 = nn.BatchNorm2d(n_filters * 8)
        self.batch_norm8_4 = nn.BatchNorm2d(n_filters * 8)
        self.batch_norm8_5 = nn.BatchNorm2d(n_filters * 8)
        self.batch_norm8_6 = nn.BatchNorm2d(n_filters * 8)
        self.batch_norm8_7 = nn.BatchNorm2d(n_filters * 8)
        self.batch_norm8_8 = nn.BatchNorm2d(n_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        #print(input.shape)
        encoder1 = self.conv1(input)
        #print(encoder1.shape)
        encoder2 = self.batch_norm2_1(self.conv2(self.leaky_relu(encoder1)))
        #print(encoder2.shape)
        encoder3 = self.batch_norm4_1(self.conv3(self.leaky_relu(encoder2)))
        #print(encoder3.shape)
        encoder4 = self.batch_norm8_1(self.conv4(self.leaky_relu(encoder3)))
        #print(encoder4.shape)
        encoder5 = self.batch_norm8_2(self.conv5(self.leaky_relu(encoder4)))
        #print(encoder5.shape)
        encoder6 = self.batch_norm8_3(self.conv6(self.leaky_relu(encoder5)))
        #print(encoder6.shape)
        encoder7 = self.batch_norm8_4(self.conv7(self.leaky_relu(encoder6)))
        #print(encoder7.shape)
        encoder8 = self.conv8(self.leaky_relu(encoder7))
        #print(encoder8.shape)
        out_f=encoder8.clone()
        out_f=torch.squeeze(out_f)

        decoder1 = self.dropout(self.batch_norm8_5(self.deconv1(self.relu(encoder8))))
        decoder1 = torch.cat((decoder1, encoder7), 1)
        #print(decoder1.shape)
        decoder2 = self.dropout(self.batch_norm8_6(self.deconv2(self.relu(decoder1))))
        decoder2 = torch.cat((decoder2, encoder6), 1)
        #print(decoder2.shape)
        decoder3 = self.dropout(self.batch_norm8_7(self.deconv3(self.relu(decoder2))))
        decoder3 = torch.cat((decoder3, encoder5), 1)
        #print(decoder3.shape)
        decoder4 = self.batch_norm8_8(self.deconv4(self.relu(decoder3)))
        decoder4 = torch.cat((decoder4, encoder4), 1)
        #print(decoder4.shape)
        decoder5 = self.batch_norm4_2(self.deconv5(self.relu(decoder4)))
        decoder5 = torch.cat((decoder5, encoder3), 1)
        #print(decoder5.shape)
        decoder6 = self.batch_norm2_2(self.deconv6(self.relu(decoder5)))
        decoder6 = torch.cat((decoder6, encoder2),1)
        #print(decoder6.shape)
        decoder7 = self.batch_norm(self.deconv7(self.relu(decoder6)))
        decoder7 = torch.cat((decoder7, encoder1), 1)
        #print(decoder7.shape)
        decoder8 = self.deconv8(self.relu(decoder7))
        #print(decoder8.shape)
        output = self.tanh(decoder8)
        #print(output.shape)
        #print("__________________________________________")
        return output,out_f

# def weights_init(m):
#     init_type='normal'
#     gain=0.02
#     classname = m.__class__.__name__
#     if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#         if init_type == 'normal':
#             init.normal_(m.weight.data, 0.0, gain)
#         elif init_type == 'xavier':
#             init.xavier_normal_(m.weight.data, gain=gain)
#         elif init_type == 'kaiming':
#             init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#         elif init_type == 'orthogonal':
#             init.orthogonal_(m.weight.data, gain=gain)
#         else:
#             raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#         if hasattr(m, 'bias') and m.bias is not None:
#             init.constant_(m.bias.data, 0.0)
#     elif classname.find('BatchNorm2d') != -1:
#         init.normal_(m.weight.data, 1.0, gain)
#         init.constant_(m.bias.data, 0.0)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


# class G(nn.Module):#U-net
#     def __init__(self,img_ch=12,output_ch=3,n_generator_filters=64):
#         super(G,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
#         self.Conv2 = conv_block(ch_in=64,ch_out=128)
#         self.Conv3 = conv_block(ch_in=128,ch_out=256)
#         self.Conv4 = conv_block(ch_in=256,ch_out=512)
#         self.Conv5 = conv_block(ch_in=512,ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4,d5),dim=1)
        
#         d5 = self.Up_conv5(d5)
        
#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1


# class G(nn.Module):#R2U_Net
#     def __init__(self,img_ch=3,output_ch=1,n_generator_filters=64,t=2):
#         super(G,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.Upsample = nn.Upsample(scale_factor=2)

#         self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

#         self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
#         self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
#         self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
#         self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.RRCNN1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.RRCNN2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.RRCNN3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.RRCNN4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.RRCNN5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4,d5),dim=1)
#         d5 = self.Up_RRCNN5(d5)
        
#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_RRCNN4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_RRCNN3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_RRCNN2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1



# class G(nn.Module):#AttU_Net
#     def __init__(self,img_ch=3,output_ch=1,n_generator_filters=64):
#         super(G,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
#         self.Conv2 = conv_block(ch_in=64,ch_out=128)
#         self.Conv3 = conv_block(ch_in=128,ch_out=256)
#         self.Conv4 = conv_block(ch_in=256,ch_out=512)
#         self.Conv5 = conv_block(ch_in=512,ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5,x=x4)
#         d5 = torch.cat((x4,d5),dim=1)        
#         d5 = self.Up_conv5(d5)
        
#         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4,x=x3)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3,x=x2)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2,x=x1)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1


# class G(nn.Module):#R2AttU_Net
#     def __init__(self,img_ch=3,output_ch=1,n_generator_filters=64,t=2):
#         super(G,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.Upsample = nn.Upsample(scale_factor=2)

#         self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

#         self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
#         self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
#         self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
#         self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
#         self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
#         self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
#         self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
#         self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.RRCNN1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.RRCNN2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.RRCNN3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.RRCNN4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.RRCNN5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5,x=x4)
#         d5 = torch.cat((x4,d5),dim=1)
#         d5 = self.Up_RRCNN5(d5)
        
#         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4,x=x3)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_RRCNN4(d4)

#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3,x=x2)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_RRCNN3(d3)

#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2,x=x1)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_RRCNN2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1

class D(nn.Module):
    def __init__(self, n_channel_input, n_channel_output, n_filters):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(n_channel_input + n_channel_output, n_filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(n_filters * 4, n_filters * 8, 4, 1, 1)
        self.conv5 = nn.Conv2d(n_filters * 8, 1, 4, 1, 1)

        self.batch_norm2 = nn.BatchNorm2d(n_filters * 2)
        self.batch_norm4 = nn.BatchNorm2d(n_filters * 4)
        self.batch_norm8 = nn.BatchNorm2d(n_filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        encoder1 = self.conv1(input)
        encoder2 = self.batch_norm2(self.conv2(self.leaky_relu(encoder1)))
        encoder3 = self.batch_norm4(self.conv3(self.leaky_relu(encoder2)))
        encoder4 = self.batch_norm8(self.conv4(self.leaky_relu(encoder3)))
        encoder5 = self.conv5(self.leaky_relu(encoder4))
        output =  self.sigmoid(encoder5)
        return output

class Fnet():
    def __init__(self):
        self.resmodel=models.resnet34(pretrained=True)
        self.resmodel.cuda()
    
    def extract_feature(self,input):
        self.resmodel.fc=torch.nn.LeakyReLU(0.1)
        self.resmodel.eval()
        
        
        #input=input.resize_(1,3,224,224)
        input=input.cuda()
                
        result=self.resmodel(Variable(input))
        result_npy=result.data
        #print(result_npy[0].shape)
        return result_npy[0]
