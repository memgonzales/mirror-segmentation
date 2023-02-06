# ===================================================
# THIS FILE CONTAINS THE MIRROR SEGMENTATION MODEL.
# EfficientNet + Proposed Boundary Extraction Module
# ===================================================

# Reference source code:
#    J. Lin, G. Wang, and R. H. Lau, "Progressive mirror detection,” in 2020
#        IEEE/CVF Conference on Computer Vision and Pattern Recognition
#        (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, June 2020,
#        pp. 3694–3702.
#    Repository: https://jiaying.link/cvpr2020-pgd/

# Mark Edward M. Gonzales & Lorene C. Uy:
# - Added annotations and comments
# - Modified the feature extraction backbone and replaced the edge detection and fusion module

import torch
import torch.nn.functional as F
from torch import nn

import timm

from backbone.resnet import resnet

# =====================================
# Convolutional block attention module
# =====================================

# Reference source code:
#    S. Woo, J. Park, J. Y. Lee, and I. S. Kweon, “CBAM: Convolutional
#        block attention module,” in Computer Vision – ECCV 2018, V. Ferrari,
#        M. Hebert, C. Sminchisescu, and Y. Weiss, Eds. Cham: Springer
#        International Publishing, 2018, pp. 3–19
#    Repository: https://github.com/Jongchan/attention-module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels

        if gate_channels // reduction_ratio > 0:
            self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
            
        else:
            self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // 4),
                nn.ReLU(),
                nn.Linear(gate_channels // 4, gate_channels)
            )
            
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw


        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
            
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)

        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# ==============================================
# Relational Contextual Contrasted Local Module
# ==============================================
class Contrast_Module_Deep(nn.Module):
    def __init__(self, planes, d1, d2):
        super(Contrast_Module_Deep, self).__init__()
        self.inplanes = int(planes)
        self.inplanes_half = int(planes / 2)
        self.outplanes = int(planes / 4)

        self.conv1 = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes_half, 3, 1, 1),
                                   nn.BatchNorm2d(self.inplanes_half), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(self.inplanes_half, self.outplanes, 3, 1, 1),
                                  nn.BatchNorm2d(self.outplanes), nn.ReLU())

        self.contrast_block_1 = Contrast_Block_Deep(self.outplanes, d1, d2)
        self.contrast_block_2 = Contrast_Block_Deep(self.outplanes,d1,d2)
        self.contrast_block_3 = Contrast_Block_Deep(self.outplanes,d1,d2)
        self.contrast_block_4 = Contrast_Block_Deep(self.outplanes,d1,d2)

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        contrast_block_1 = self.contrast_block_1(conv2)
        contrast_block_2 = self.contrast_block_2(contrast_block_1)
        contrast_block_3 = self.contrast_block_3(contrast_block_2)
        contrast_block_4 = self.contrast_block_4(contrast_block_3)

        output = self.cbam(torch.cat((contrast_block_1, contrast_block_2, contrast_block_3, contrast_block_4), 1))

        return output



class Contrast_Block_Deep(nn.Module):
    def __init__(self, planes, d1, d2):
        super(Contrast_Block_Deep, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 2)

        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d1, dilation=d1)

        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)

        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)


        self.relu = nn.ReLU()

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        local_1 = self.local_1(x)
        context_1 = self.context_1(x)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn1(ccl_1)
        ccl_1 = self.relu(ccl_1)

        local_2 = self.local_2(x)
        context_2 = self.context_2(x)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn2(ccl_2)
        ccl_2 = self.relu(ccl_2)

        output = self.cbam(torch.cat((ccl_1, ccl_2), 1))

        return output




class Resudial_Block(nn.Module):
    def __init__(self, in_c):
        super(Resudial_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = self.relu(x)
        return x



# ==================
# Refinement module
# ==================
class Refinement_Net(nn.Module):
    def __init__(self, in_c):
        super(Refinement_Net, self).__init__()
        self.conv1 = BasicConv(in_planes=in_c, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)

        self.res1 = Resudial_Block(64)
        self.res2 = Resudial_Block(64)
        self.res3 = Resudial_Block(64)

        self.final_conv = nn.Conv2d(64 + 1, 1, 3, 1, 1)

    def forward(self, image, saliency_map, edge):
        fusion = torch.cat((edge, saliency_map, image), 1)
        fusion = self.conv1(fusion)
        fusion = self.conv2(fusion)
        fusion = self.res1(fusion)
        fusion = self.res2(fusion)
        fusion = self.res3(fusion)
        fusion = self.final_conv(torch.cat((saliency_map, fusion), 1))
        return fusion


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

# =============================
# Criss-Cross Attention Module
# =============================

# Reference source code:
#   Z. Huang, X. Wang, L. Huang, C. Huang, Y. Wei and W. Liu, 
#       "CCNet: Criss-Cross Attention for Semantic Segmentation," 
#      2019 IEEE/CVF International Conference on Computer Vision (ICCV), 
#      2019, pp. 603-612, doi: 10.1109/ICCV.2019.00069.
#   Repository: https://github.com/Serge-weihao/CCNet-Pure-Pytorch
class RAttention(nn.Module):
    def __init__(self,in_dim):
        super(RAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_query_LR = torch.diagonal(proj_query, 0, 2, 3)
        proj_query_RL = torch.diagonal(torch.transpose(proj_query, 2, 3), 0, 2, 3)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        proj_key_LR = torch.diagonal(proj_key, 0, 2, 3).permute(0,2,1).contiguous()
        proj_key_RL = torch.diagonal(torch.transpose(proj_key, 2, 3), 0, 2, 3).permute(0,2,1).contiguous()

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        proj_value_LR = torch.diagonal(proj_value, 0, 2, 3)
        proj_value_RL = torch.diagonal(torch.transpose(proj_value, 2, 3), 0, 2, 3)
        
        
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)

        energy_LR = torch.bmm(proj_key_LR, proj_query_LR)
        energy_RL = torch.bmm(proj_key_RL, proj_query_RL)


        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        out_LR = self.softmax(torch.bmm(proj_value_LR, energy_LR).unsqueeze(-1))
        out_RL = self.softmax(torch.bmm(proj_value_RL, energy_RL).unsqueeze(-1))

        return self.gamma*(out_H + out_W + out_LR + out_RL) + x

class Relation_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Relation_Attention, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.ra = RAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

            
    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.ra(output)
        output = self.convb(output)
        
        return output
        
# ===================================================
# Proposed Boundary Extraction and Prediction Module
# ===================================================
class BFE_Module(nn.Module):
    def __init__(self, planes):
        super(BFE_Module, self).__init__()
        self.inplanes = 56
        self.inplanes_half = 28

        self.edge_layer1 = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes_half, 1, 1, dilation = 1),
                                   nn.BatchNorm2d(self.inplanes_half), nn.ReLU())
        self.edge_layer2 = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes_half, 3, 1, dilation = 1),
                                   nn.BatchNorm2d(self.inplanes_half), nn.ReLU())
        self.edge_layer3 = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes_half, 3, 1, dilation = 2),
                                   nn.BatchNorm2d(self.inplanes_half), nn.ReLU())
        self.edge_layer4 = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes_half, 3, 1, dilation = 4),
                                   nn.BatchNorm2d(self.inplanes_half), nn.ReLU())

        self.cbam = CBAM(self.inplanes)                                   

    def forward(self, x):
        conv1 = self.edge_layer1(x)
        conv2 = self.edge_layer2(x)
        conv2 = F.upsample(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv3 = self.edge_layer3(x)
        conv3 = F.upsample(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)

        conv4 = self.edge_layer4(x)
        conv4 = F.upsample(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)

        output = torch.cat((conv1, conv2, conv3, conv4), 1)

        return output


# ========
# Network
# ========
class PMDLite(nn.Module):
    def __init__(self, training=False):
        super(PMDLite, self).__init__()
        self.m = timm.create_model('efficientnetv2_rw_m', features_only=True, pretrained=True)
        
        self.edge_extract = BFE_Module(2048)
        
        self.edge_predict = nn.Sequential(nn.Conv2d(194, 97, 1, 1, 1), nn.BatchNorm2d(97),
                                nn.ReLU(), nn.Conv2d(97, 1, 3, 1, 1))

        self.contrast_4 = Contrast_Module_Deep(328,d1=2, d2=4)
        self.contrast_3 = Contrast_Module_Deep(192,d1=4, d2=8)
        self.contrast_2 = Contrast_Module_Deep(80, d1=4, d2=8)
        self.contrast_1 = Contrast_Module_Deep(56, d1=4, d2=8)

        self.ra_4 = Relation_Attention(328, 328)
        self.ra_3 = Relation_Attention(192, 192)
        self.ra_2 = Relation_Attention(80, 80)
        self.ra_1 = Relation_Attention(56, 56)

        self.up_4 = nn.Sequential(nn.ConvTranspose2d(328, 82, 4, 2, 1), nn.BatchNorm2d(82), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(192, 48, 4, 2, 1), nn.BatchNorm2d(48), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(80, 20, 4, 2, 1), nn.BatchNorm2d(20), nn.ReLU())
        self.up_1 = nn.Sequential(nn.ConvTranspose2d(56, 14, 4, 2, 1), nn.BatchNorm2d(14), nn.ReLU())

        self.cbam_4 = CBAM(82)
        self.cbam_3 = CBAM(48)
        self.cbam_2 = CBAM(20)
        self.cbam_1 = CBAM(14)

        self.layer4_predict = nn.Conv2d(82, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(48, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(20, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(14, 1, 3, 1, 1)

        self.refinement = nn.Conv2d(1+1+3+1+1+1, 1, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        efficient_net = self.m(x)
        layer0 = efficient_net[0]
        layer1 = efficient_net[1]
        layer2 = efficient_net[2]
        layer3 = efficient_net[3]
        layer4 = efficient_net[4]

        contrast_4 = self.contrast_4(layer4)
        cc_att_map_4 = self.ra_4(layer4)
        final_contrast_4 = contrast_4 * cc_att_map_4

        up_4 = self.up_4(final_contrast_4)
        cbam_4 = self.cbam_4(up_4)
        layer4_predict = self.layer4_predict(cbam_4)
        layer4_map = F.sigmoid(layer4_predict)

        contrast_3 = self.contrast_3(layer3 * layer4_map)
        cc_att_map_3 = self.ra_3(layer3 * layer4_map)

        final_contrast_3 = contrast_3 * cc_att_map_3

        up_3 = self.up_3(final_contrast_3)
        cbam_3 = self.cbam_3(up_3)
        layer3_predict = self.layer3_predict(cbam_3)
        layer3_map = F.sigmoid(layer3_predict)
        
        contrast_2 = self.contrast_2(layer2 * layer3_map)
        cc_att_map_2 = self.ra_2(layer2 * layer3_map)
        final_contrast_2 = contrast_2 * cc_att_map_2

        up_2 = self.up_2(final_contrast_2)
        cbam_2 = self.cbam_2(up_2)
        layer2_predict = self.layer2_predict(cbam_2)
        layer2_map = F.sigmoid(layer2_predict)

        contrast_1 = self.contrast_1(layer1 * layer2_map)
        cc_att_map_1 = self.ra_1(layer1 * layer2_map)
        final_contrast_1 = contrast_1 * cc_att_map_1

        up_1 = self.up_1(final_contrast_1)
        cbam_1 = self.cbam_1(up_1)
        layer1_predict = self.layer1_predict(cbam_1)

        edge_feature = self.edge_extract(layer1)
        layer4_edge_feature = F.upsample(cbam_4, size=edge_feature.size()[2:], mode='bilinear', align_corners=True)
        
        final_edge_feature = torch.cat( (edge_feature, layer4_edge_feature), 1)
        
        layer0_edge = self.edge_predict(final_edge_feature)
        

        layer4_predict = F.upsample(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)


        layer0_edge = F.upsample(layer0_edge, size=x.size()[2:], mode='bilinear', align_corners=True)

        final_features = torch.cat((x, layer1_predict, layer0_edge, layer2_predict, layer3_predict, layer4_predict),1)
        final_predict = self.refinement(final_features)
        final_predict = F.upsample(final_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_edge, final_predict

        return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
               F.sigmoid(layer1_predict), F.sigmoid(layer0_edge), F.sigmoid(final_predict)
