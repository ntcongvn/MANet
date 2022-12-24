import sys

from net.layer import *

from config import net_config as config
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time
import torch.nn.functional as F
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm


bn_momentum = 0.1
affine = True
pdrop=0.1
relu_inplace=True
drop_inplace=False


#Projection module (PM)
class ContextProjection3d(nn.Module):                        
    def __init__(self, n_in, n_out, stride = 1):
        super(ContextProjection3d, self).__init__()
        
        self.conv1_hw=nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.conv1_dh=nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.conv1_dw=nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.conv1_dhw=nn.Conv3d(1, 1, kernel_size = 3, stride = stride, padding = 1)

        self.bn1_hw = nn.BatchNorm2d(n_out, momentum=bn_momentum)
        self.bn1_dh = nn.BatchNorm2d(n_out, momentum=bn_momentum)
        self.bn1_dw = nn.BatchNorm2d(n_out, momentum=bn_momentum)
        self.bn1_dhw = nn.BatchNorm3d(1, momentum=bn_momentum)

        self.conv2_hw=nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.conv2_dh=nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.conv2_dw=nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.conv2_dhw=nn.Conv3d(1, 1, kernel_size = 3, padding = 1)
        
        self.bn2_hw = nn.BatchNorm2d(n_out, momentum=bn_momentum)
        self.bn2_dh = nn.BatchNorm2d(n_out, momentum=bn_momentum)
        self.bn2_dw = nn.BatchNorm2d(n_out, momentum=bn_momentum)
        self.bn2_dhw = nn.BatchNorm3d(1, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace = relu_inplace)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):      #batch x chanel x D x H x W
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        out_dhw=torch.mean(x,1,keepdim=True)
        out_dhw = self.conv1_dhw(out_dhw)
        out_dhw = self.bn1_dhw(out_dhw)
        out_dhw = self.relu(out_dhw)
        out_dhw = self.conv2_dhw(out_dhw)
        out_dhw = self.bn2_dhw(out_dhw)

        out_hw=torch.mean(x,2)
        out_hw = self.conv1_hw(out_hw)
        out_hw = self.bn1_hw(out_hw)
        out_hw = self.relu(out_hw)
        out_hw = self.conv2_hw(out_hw)
        out_hw = self.bn2_hw(out_hw)

        out_dw=torch.mean(x,3)
        out_dw = self.conv1_dw(out_dw)
        out_dw = self.bn1_dw(out_dw)
        out_dw = self.relu(out_dw)
        out_dw = self.conv2_dw(out_dw)
        out_dw = self.bn2_dw(out_dw)

        out_dh=torch.mean(x,4)
        out_dh = self.conv1_dh(out_dh)
        out_dh = self.bn1_dh(out_dh)
        out_dh = self.relu(out_dh)
        out_dh = self.conv2_dh(out_dh)
        out_dh = self.bn2_dh(out_dh)
        
        C=out_hw.size()[1]
        H=out_hw.size()[2]
        W=out_hw.size()[3]
        D=out_dh.size()[2]
        out_dhw=out_dhw.expand(-1,C,-1,-1,-1)
        out_hw=torch.unsqueeze(out_hw, 2).expand(-1,-1,D,-1,-1)
        out_dw=torch.unsqueeze(out_dw, 3).expand(-1,-1,-1,H,-1)
        out_dh=torch.unsqueeze(out_dh, 4).expand(-1,-1,-1,-1,W)
        out =out_dhw+out_hw+out_hw+out_hw
        #out=out*x
        out += residual
        out = self.relu(out)
        return out

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = relu_inplace)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace = relu_inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,stride=stride, padding=padding)
        self.dropout = nn.Dropout3d(p=0.1)
        self.conv2 = nn.Conv3d(in_channels//4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

#Boundary enhancement module (BEM)
class BEM(nn.Module):
    def __init__(self):
        super(BEM, self).__init__()
        self.relu = nn.ReLU(inplace = relu_inplace)
    def forward(self, x, pred):
        residual = x
        score=F.interpolate(pred, scale_factor=2, mode='trilinear', align_corners=True,recompute_scale_factor= False)
        score = torch.sigmoid(score)     
        
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        #att=torch.abs(1 - score)

        att_x = x * att
        out = att_x + residual
        out = self.relu(out)
        return out

#Fast cascading context module (FCM)
class CCM(nn.Module):
    def __init__(self, in_channels, out_channels,pool_size = [1, 3, 5],in_channel_list=[],out_channel_list = [256, 128/2],cascade=False):
        super(CCM, self).__init__()
        self.cascade=cascade
        self.in_channel_list=in_channel_list
        self.out_channel_list = out_channel_list
        upsampe_scale = [2,4,8,16]
        GClist = []
        GCoutlist = []

        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveMaxPool3d(ps),
                nn.Conv3d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        self.GCmodule = nn.ModuleList(GClist)
        self.synthetic=nn.Sequential(nn.Conv3d(out_channels * 3, in_channels, 3, 1, 1),
                                            nn.BatchNorm3d(in_channels),
                                            nn.ReLU(inplace=True))

        

        for i in range(len(self.out_channel_list)):
            GCoutlist.append(nn.Sequential(nn.Conv3d(in_channels, self.out_channel_list[i], 3, 1, 1),
                                          nn.BatchNorm3d(self.out_channel_list[i]),
                                          nn.ReLU(inplace=True),
                                          nn.Upsample(scale_factor=upsampe_scale[i],mode='trilinear',align_corners=True)))
        self.GCoutmodel = nn.ModuleList(GCoutlist)


    def forward(self, x,y=None):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule)):
          global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='trilinear', align_corners=True))
        global_context = torch.cat(global_context, dim=1)
        global_context=self.synthetic(global_context)
        
        if self.cascade is True and y is not None:
          global_context=global_context+y
        
        output = []
        for i in range(len(self.GCoutmodel)):
            out=self.GCoutmodel[i](global_context) 
            output.append(out)
        return output

class FeatureNet(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size = 3, padding = 1,stride=2),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = relu_inplace),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = relu_inplace)
            )

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            )

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 64),
            )

        self.forw3 = nn.Sequential(
            ResBlock3d(64, 128),
            )

        self.forw4 = nn.Sequential(
            ResBlock3d(128, 256),
            )

        self.back1 = nn.Sequential(
            ResBlock3d(96+32+32+32, 48),
            )

        self.back2 = nn.Sequential(
            ResBlock3d(192+64+64+64, 96),
            )

        self.back3 = nn.Sequential(
            ResBlock3d(256+128+128, 192),
            )

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
            )

        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(192, 192, kernel_size=2, stride=2),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True)
            )

        self.path3 = nn.Sequential(
            nn.ConvTranspose3d(96, 96, kernel_size=2, stride=2),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True)
            )
      
        self.sideout1 = SideoutBlock(48, 1)
        self.sideout2 = SideoutBlock(96, 1)
        self.sideout3 = SideoutBlock(192, 1)

        self.contextproj1 = ContextProjection3d(32,32)
        self.contextproj2 = ContextProjection3d(64,64)
        self.contextproj3 = ContextProjection3d(128,128)
        
        # cascade context module  #64 32 16 16
        self.ccm4 = CCM(256, 256//3,pool_size = [1, 3, 5],in_channel_list = [],out_channel_list = [128])
        self.ccm3 = CCM(128, 128//3,pool_size = [2, 6, 10],in_channel_list = [128],out_channel_list = [64],cascade=True)
        self.ccm2 = CCM(64, 64//3,pool_size = [4, 12, 20],in_channel_list = [64],out_channel_list = [32],cascade=True)

        self.guidanceblock=BEM()


                                                                        
    def forward(self, x):                                         
        out = self.preBlock(x)#16                                 
        out_pool=out
        
        out1 = self.forw1(out_pool)#32                            
        out1_pool, _ = self.maxpool2(out1)                        
        
        out2 = self.forw2(out1_pool)#64                           
        out2_pool, _ = self.maxpool3(out2)                        
        
        out3 = self.forw3(out2_pool)#96                           
        out3_pool, _ = self.maxpool4(out3)                        
      
        out4 = self.forw4(out3_pool)#96                           


        cascade_context4=self.ccm4(out4)
        cascade_context3=self.ccm3(out3,cascade_context4[0])
        cascade_context2=self.ccm2(out2,cascade_context3[0])


        rev3 = self.path1(out4)                                   
        contextproj3=self.contextproj3(out3)
        comb3 = self.back3(torch.cat((rev3, contextproj3,cascade_context4[0]), 1))    
        
        sideout3=self.sideout3(comb3)
        rev2 = self.path2(comb3)                                  
        gb2=self.guidanceblock(out2,sideout3)
        contextproj2=self.contextproj2(out2)
        comb2 = self.back2(torch.cat((rev2, contextproj2,gb2,cascade_context3[0]), 1))            
        sideout2=self.sideout2(comb2)
        
        rev1 = self.path3(comb2)                                 
        gb1=self.guidanceblock(out1,sideout2)
        contextproj1=self.contextproj1(out1)
        comb1= self.back1(torch.cat((rev1, contextproj1,gb1,cascade_context2[0]), 1))     
        sideout1=self.sideout1(comb1)

        
        comb0= comb1
        sideout0=comb0
        return [x,out,out1,out2,out3,comb0, comb1, comb2,comb3,sideout0,sideout1,sideout2,sideout3],[sideout0,sideout1,sideout2,sideout3]

class RpnHead(nn.Module):
    def __init__(self, config, in_channels=96):
        super(RpnHead, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 32, kernel_size=1),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(inplace = relu_inplace),
                                    nn.Dropout3d(p=0.1))
        self.logits = nn.Conv3d(32, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(32, 6 * len(config['anchors']), kernel_size=3,padding = 1)

    def forward(self, f):
        out = self.conv(f)                
        logits = self.logits(out)         
        deltas = self.deltas(out)         

        size = logits.size()
        
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 1)
        
        size = deltas.size()
        deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
        deltas = deltas.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 6)
        
        return logits, deltas 

class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels=64+96,mid_channels=32):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.conv = nn.Conv3d(in_channels, mid_channels,kernel_size=1)
        self.conv_bn = nn.BatchNorm3d(mid_channels)
        self.conv_relu = nn.ReLU(inplace = relu_inplace)
        
        self.fc1 = nn.Linear(mid_channels * self.crop_size[0] * self.crop_size[1] * self.crop_size[2], 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc1_dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc2_dropout = nn.Dropout(0.5)
        self.logit = nn.Linear(128, self.num_class)
        self.delta = nn.Linear(128, self.num_class * 6)

    def forward(self, crops):
        x=self.conv(crops)
        x=self.conv_bn(x)
        x=self.conv_relu(x)    

        x = x.view(x.size(0), -1)                       
        
        x=self.fc1(x)
        x=self.fc1_bn(x)
        x = F.relu(x,inplace = relu_inplace)                         
        x=self.fc1_dropout(x)
        
        x=self.fc2(x)
        x=self.fc2_bn(x)
        x = F.relu(x,inplace = relu_inplace)               
        x=self.fc2_dropout(x)

        logits = self.logit(x)                 
        deltas = self.delta(x)                 
        return logits, deltas

def crop_mask_regions(masks, crop_boxes):
    out = []
    for i in range(len(crop_boxes)):
        b, z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]
        m = masks[i][z_start:z_end, y_start:y_end, x_start:x_end].contiguous()
        out.append(m)
    return out

def top1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        res.append(preds[0])
        
    res = np.array(res)
    return res

def random1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        idx = random.sample(range(len(preds)), 1)[0]
        res.append(preds[idx])
        
    res = np.array(res)
    return res

class CropRoi(nn.Module):
    def __init__(self, cfg, rcnn_crop_size):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size  = cfg['rcnn_crop_size']

    def forward(self, f, inputs, proposals,scale):
        DEPTH, HEIGHT, WIDTH = inputs.shape[2:]

        crops = []
        for p in proposals:
            b = int(p[0])
            center = p[2:5]
            side_length = p[5:8]
            c0 = center - side_length / 2 
            c1 = c0 + side_length # 
            c0 = (c0 / scale).floor().long()
            c1 = (c1 / scale).ceil().long()
            minimum = torch.LongTensor([[0, 0, 0]]).cuda()
            maximum = torch.LongTensor(
                np.array([[DEPTH, HEIGHT, WIDTH]]) / scale).cuda()

            c0 = torch.cat((c0.unsqueeze(0), minimum), 0)
            c1 = torch.cat((c1.unsqueeze(0), maximum), 0)
            c0, _ = torch.max(c0, 0)
            c1, _ = torch.min(c1, 0)

            if np.any((c1 - c0).cpu().data.numpy() < 1):
                print(p)
                print('c0:', c0, ', c1:', c1)
            crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops

class MaskHead(nn.Module):
    def __init__(self, cfg,mid_chanel=24):
        super(MaskHead, self).__init__()
        self.num_class = cfg['num_class']


        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True)
          )
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear',align_corners=True),
          )
        self.upsample8 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='trilinear',align_corners=True),
          )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False),
          )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear',align_corners=False),
          )

        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='trilinear',align_corners=False),
          )

                
        self.next2 = nn.Sequential(
            nn.Conv3d(32+48, mid_chanel, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm3d(mid_chanel, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
       
        self.next4 = nn.Sequential(
            nn.Conv3d(64+96, mid_chanel, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm3d(mid_chanel, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        self.next8 = nn.Sequential(
            nn.Conv3d(128+192, mid_chanel, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm3d(mid_chanel, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.nextout = nn.Sequential(
            nn.Conv3d(mid_chanel*3+1,mid_chanel, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm3d(mid_chanel, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True),
            nn.Dropout3d(0.1))

        for i in range(self.num_class):
            setattr(self, 'logits' + str(i + 1), nn.Conv3d(mid_chanel, 1, kernel_size=1))

    def forward(self, detections, features):

        img,out_1,out_2,out_4,out_8,f_1, f_2, f_4,f_8,sideout_1,sideout_2,sideout_4,sideout_8=features                      #feature_scale
        # Squeeze the first dimension to recover from protection on avoiding split by dataparallel      
        img = img.squeeze(0)
        out_1 = out_1.squeeze(0)
        out_2 = out_2.squeeze(0)
        out_4 = out_4.squeeze(0)
        out_8 = out_8.squeeze(0)
        f_2 = f_2.squeeze(0)
        f_4 = f_4.squeeze(0)
        f_8 = f_8.squeeze(0)
    
        _, _, D, H, W = img.shape
        out = []

        for detection in detections:
            b, z_start, y_start, x_start, z_end, y_end, x_end, cat = detection

            o8 = out_8[b, :, z_start // 8:z_end // 8, y_start // 8:y_end // 8, x_start // 8:x_end // 8].unsqueeze(0)
            f8 = f_8[b, :, z_start // 8:z_end // 8, y_start // 8:y_end // 8, x_start // 8:x_end // 8].unsqueeze(0)
            

            o4 = out_4[b, :, z_start // 4:z_end // 4, y_start // 4:y_end // 4, x_start // 4:x_end // 4].unsqueeze(0)
            f4 = f_4[b, :, z_start // 4:z_end // 4, y_start // 4:y_end // 4, x_start // 4:x_end // 4].unsqueeze(0)
            

            o2 = out_2[b, :, z_start // 2:z_end // 2, y_start // 2:y_end // 2, x_start // 2:x_end // 2].unsqueeze(0)
            f2 = f_2[b, :, z_start // 2:z_end // 2, y_start // 2:y_end // 2, x_start // 2:x_end // 2].unsqueeze(0)
            
            im = img[b, :, z_start:z_end, y_start:y_end, x_start:x_end].unsqueeze(0)

            #Encoder branch
  
            of2=self.next2(torch.cat((o2,f2),1))
            of4=self.next4(torch.cat((o4,f4),1))
            of8=self.next8(torch.cat((self.up2(o8),self.up2(f8)),1))

            x=torch.cat((im,self.up2(of2),self.up4(of4),self.up4(of8)),1)
            x=self.nextout(x)

            #Decoder branch
            logits = getattr(self, 'logits' + str(int(cat)))(x)
            logits = logits.squeeze()

            #mask = Variable(torch.zeros((D, H, W))).cuda()
            #Exponentail memory
            mask = Variable(torch.zeros((D, H, W))).cpu()
            mask[z_start:z_end, y_start:y_end, x_start:x_end] = logits
            mask = mask.unsqueeze(0)
            out.append(mask)

        out=torch.cat(out, 0)
        #Exponentail memory
        out=out.cuda()
        return out

class MANet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(MANet, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.feature_net = FeatureNet(config, 1, 64)
        self.rpn = RpnHead(config)
        self.rcnn_head = RcnnHead(config)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.mask_head = MaskHead(config)
        self.use_rcnn = False
        self.use_mask = False

    def forward(self, inputs, truth_boxes, truth_labels, truth_masks, masks, split_combiner=None, nzhw=None):
        features,ds_predict = data_parallel(self.feature_net, (inputs)); 
        self.ds=[ds_predict,masks]

        fs = features[7]
        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)
        b,D,H,W,_,num_class = self.rpn_logits_flat.shape
        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1);
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6);


        self.rpn_window    = make_rpn_windows(fs, self.cfg)
        self.rpn_proposals = []
        if self.use_rcnn or self.mode in ['eval', 'test']:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat)

        if self.mode in ['train', 'valid']:
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(self.cfg, self.mode, inputs, self.rpn_window, truth_boxes, truth_labels )

            if self.use_rcnn:
                self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                    make_rcnn_target(self.cfg, self.mode, inputs, self.rpn_proposals,
                        truth_boxes, truth_labels, truth_masks)

        #rcnn proposals
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.ensemble_proposals = copy.deepcopy(self.rpn_proposals)

        self.mask_probs = []
        if self.use_rcnn:
            if len(self.rpn_proposals) > 0:

                rcnn_crops_e2 = self.rcnn_crop(features[3], inputs, self.rpn_proposals,self.cfg['stride'])      
                rcnn_crops_d2 = self.rcnn_crop(features[7], inputs, self.rpn_proposals,self.cfg['stride'])

                rcnn_crops=torch.cat((rcnn_crops_e2,rcnn_crops_d2), 1)
                self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
                self.detections, self.keeps = rcnn_nms(self.cfg, self.mode, inputs, self.rpn_proposals, 
                                                                        self.rcnn_logits, self.rcnn_deltas)

            if self.mode in ['eval']:
                # Ensemble
                fpr_res = get_probability(self.cfg, self.mode, inputs, self.rpn_proposals,  self.rcnn_logits, self.rcnn_deltas)
                self.ensemble_proposals[:, 1] = (self.ensemble_proposals[:, 1] + fpr_res[:, 0]) / 2

            if self.use_mask and len(self.detections):
                # keep batch index, z, y, x, d, h, w, class
                self.crop_boxes = []
                if len(self.detections):
                    self.crop_boxes = self.detections[:, [0, 2, 3, 4, 5, 6, 7, 8]].cpu().numpy().copy()
                    self.crop_boxes[:, 1:-1] = center_box_to_coord_box(self.crop_boxes[:, 1:-1])
                    self.crop_boxes = self.crop_boxes.astype(np.int32)

                    #backup
                    self.crop_boxes_origin = copy.deepcopy(self.crop_boxes)
                    self.crop_boxes_origin[:, 1:-1] = clip_boxes(self.crop_boxes_origin[:, 1:-1], inputs.shape[2:])

                    self.crop_boxes[:, 1:-1] = ext2factor(self.crop_boxes[:, 1:-1], 8)
                    self.crop_boxes[:, 1:-1] = clip_boxes(self.crop_boxes[:, 1:-1], inputs.shape[2:])

                if self.mode in ['train', 'valid']:
                    self.mask_targets = make_mask_target(self.cfg, self.mode, inputs, self.crop_boxes,
                        truth_boxes, truth_labels, masks)

                # Make sure to keep feature maps not splitted by data parallel
                
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
                self.mask_probs = data_parallel(self.mask_head, (torch.from_numpy(self.crop_boxes).cuda(), features))
                

                if self.mode in ['eval', 'test']:
                    mask_keep = mask_nms(self.cfg, self.mode, self.mask_probs, self.crop_boxes, inputs)

                    self.crop_boxes =self.crop_boxes_origin[mask_keep] #self.crop_boxes[mask_keep]
                    self.detections = self.detections[mask_keep]
                    ##Exponentail memory
                    self.mask_probs = self.mask_probs.cpu()
                    self.mask_probs = self.mask_probs[mask_keep]
                    self.mask_probs = self.mask_probs.cuda()
                    ##self.mask_probs = self.mask_probs[mask_keep]
                
                self.mask_probs = crop_mask_regions(self.mask_probs, self.crop_boxes)



    def loss(self, targets=None):
        cfg  = self.cfg
    
        self.rcnn_cls_loss, self.rcnn_reg_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        rcnn_stats = None
        mask_stats = None

        self.mask_loss = torch.zeros(1).cuda()
    
        self.rpn_cls_loss, self.rpn_reg_loss, rpn_stats = \
           rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels,
            self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights, self.cfg, mode=self.mode)
    
        if self.use_rcnn:
            self.rcnn_cls_loss, self.rcnn_reg_loss, rcnn_stats = \
                rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets)

        if self.use_mask:
            self.mask_loss, mask_losses = mask_loss(self.mask_probs, self.mask_targets)
            mask_stats = [[] for _ in range(cfg['num_class'] - 1)] 
            for i in range(len(self.crop_boxes)):
                cat = int(self.crop_boxes[i][-1]) - 1
                mask_stats[cat].append(mask_losses[i])
            mask_stats = [np.mean(e) for e in mask_stats]
            mask_stats = np.array(mask_stats)
            mask_stats[mask_stats == 0] = np.nan
    
        #Deep supervision
        ds_predicts,ds_mask=self.ds
        ds_mask=torch.from_numpy(np.array(ds_mask)).cuda()
        self.ds_loss=DeepSupervisionLoss(ds_predicts,ds_mask)

        self.total_loss =self.ds_loss + self.rpn_cls_loss + self.rpn_reg_loss \
                          + self.rcnn_cls_loss +  self.rcnn_reg_loss \
                          + self.mask_loss

    
        return self.total_loss, rpn_stats, rcnn_stats, mask_stats,self.ds_loss

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()

    def set_anchor_params(self, anchor_ids, anchor_params):
        self.anchor_ids = anchor_ids
        self.anchor_params = anchor_params

    def crf(self, detections):
        """
        detections: numpy array of detection results [b, z, y, x, d, h, w, p]
        """
        res = []
        config = self.cfg
        anchor_ids = self.anchor_ids
        anchor_params = self.anchor_params
        anchor_centers = []

        for a in anchor_ids:
            # category starts from 1 with 0 denoting background
            # id starts from 0
            cat = a + 1
            dets = detections[detections[:, -1] == cat]
            if len(dets):
                b, p, z, y, x, d, h, w, _ = dets[0]
                anchor_centers.append([z, y, x])
                res.append(dets[0])
            else:
                # Does not have anchor box
                return detections
        
        pred_cats = np.unique(detections[:, -1]).astype(np.uint8)
        for cat in pred_cats:
            if cat - 1 not in anchor_ids:
                cat = int(cat)
                preds = detections[detections[:, -1] == cat]
                score = np.zeros((len(preds),))
                roi_name = config['roi_names'][cat - 1]

                for k, params in enumerate(anchor_params):
                    param = params[roi_name]
                    for i, det in enumerate(preds):
                        b, p, z, y, x, d, h, w, _ = det
                        d = np.array([z, y, x]) - np.array(anchor_centers[k])
                        prob = norm.pdf(d, param[0], param[1])
                        prob = np.log(prob)
                        prob = np.sum(prob)
                        score[i] += prob

                res.append(preds[score == score.max()][0])
            
        res = np.array(res)
        return res

if __name__ == '__main__':
    net = FasterRcnn(config)

    input = torch.rand([4,1,128,128,128])
    input = Variable(input)
