# ================================================================
# THIS FILE IMPLEMENTS THE LOSS FUNCTIONS USED IN MODEL TRAINING.
# ================================================================

# Reference Source Codes:
#    H. Mei, G. P. Ji, Z. Wei, X. Yang, X. Wei, and D. P. Fang (2021). 
#        "Camouflaged object segmentation with distraction mining," 
#        in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 
#        Nashville, TN, USA: IEEE Computer Society, June 2021, pp. 8768–8877.
#    Repository: https://github.com/Mhaiyang/CVPR2021_PFNet
#
#    [For additive loss combining weighted BCE and IoU loss]
#    J. Wei, S. Wang, and Q. Huang, "F³net: Fusion, feedback and focus
#        for salient object detection," Proceedings of the AAAI Conference on
#        Artificial Intelligence, vol. 34, no. 07, pp. 12321–12328, Apr. 2020.
#    Repository: https://github.com/weijun88/F3Net/blob/master/src/train.py


# Mark Edward M. Gonzales & Lorene C. Uy:
# - Added annotations and comments

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Intersection-over-union loss
# =============================
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)

# ==========================
# Binary cross-entropy loss
# ==========================
def cross_entropy(logits, labels):
    return torch.mean((1 - labels) * logits + torch.log(1 + torch.exp(-logits)))

# ==================================================
# Additive loss combining weighted BCE and IoU loss
# ==================================================
# Proposed in:
#    J. Wei, S. Wang, and Q. Huang, "F³net: Fusion, feedback and focus
#        for salient object detection," Proceedings of the AAAI Conference on
#        Artificial Intelligence, vol. 34, no. 07, pp. 12 321–12 328, Apr. 2020.
class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)

# ==========================================
# Laplacian-based loss for edge enhancement
# ==========================================
# Proposed in:
#    T. Zhao and X. Wu, "Pyramid feature attention network for saliency
#        detection," in 2019 IEEE/CVF Conference on Computer Vision and
#        Pattern Recognition (CVPR), 2019, pp. 3080–3089.
class edge_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # Match with the filter shape in Pytorch: out_channel, in_channel, height, width.
        laplace = torch.FloatTensor([[-1,-1,-1,],[-1,8,-1],[-1,-1,-1]]).view([1,1,3,3])
        self.laplace = nn.Parameter(data=laplace, requires_grad=False)

    # Get the Laplacian, which is related to edge enhancement.
    def torchLaplace(self, x):
        edge = F.conv2d(x, self.laplace, padding=1)
        edge = torch.abs(torch.tanh(edge))
        return edge

    def forward(self, y_pred, y_true, mode=None):
        y_true_edge = self.torchLaplace(y_true)
        y_pred_edge = self.torchLaplace(y_pred)
        edge_loss = cross_entropy(y_pred_edge, y_true_edge)
        
        return edge_loss
