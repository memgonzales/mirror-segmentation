# ==================================================================
# THIS FILE CONTAINS THE METHOD FOR EVALUATING THE PREDICTED MASKS.
# ==================================================================

# Reference source code:
#    J. Lin, G. Wang, and R. H. Lau, "Progressive mirror detection,” in 2020
#        IEEE/CVF Conference on Computer Vision and Pattern Recognition
#        (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, June 2020,
#        pp. 3694–3702.
#    Repository: https://jiaying.link/cvpr2020-pgd/

# Mark Edward M. Gonzales & Lorene C. Uy:
# - Added annotations and comments

from typing import OrderedDict
import numpy as np
import os
from PIL import Image
import skimage.io
import skimage.transform

import pydensecrf.densecrf as dcrf

from config import dataset_name, result_path, testing_path

# =============================
# Class for taking the average
# =============================
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ===========================================
# Create a directory if it doesn't exist yet.
# ===========================================
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ====================
# Compute the sigmoid.
# ====================
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Use a fully connected conditional random field for post-processing.
# Proposed in:
#    P. Krahenb ¨ uhl and V. Koltun, “Efficient inference in fully connected ¨
#        CRFs with Gaussian edge potentials,” in Advances in Neural Information
#        Processing Systems, J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira,
#        and K. Weinberger, Eds., vol. 24. Curran Associates, Inc., 2011.
def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    
    # Setup the fully connected conditional random field.
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Perform inference.
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')

# ===================================
# Return the size of the mirror mask.
# ===================================
def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

# ==========================================================
# Check if the mask and the ground truth have the same size.
# ==========================================================
def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

# ==========================
# Get the ground truth mask.
# ==========================
def get_gt_mask(imgname, MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    mask = skimage.io.imread(mask_path)
    mask = np.where(mask == 255, 1, 0).astype(np.float32)

    return mask

# ==================================
# Get the normalized predicted mask.
# ==================================
def get_normalized_predict_mask(imgname, PREDICT_MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = PREDICT_MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    if not os.path.exists(mask_path):
        print("{} has no predict mask!".format(imgname))
    mask = skimage.io.imread(mask_path).astype(np.float32)
    if np.max(mask) > 0:
        mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask))
    mask = mask.astype(np.float32)
    mask = skimage.color.rgb2grey(mask)

    return mask

# ==============================
# Get the binary predicted mask.
# ==============================
def get_binary_predict_mask(imgname, PREDICT_MASK_DIR):
    filestr = imgname[:-4]
    mask_folder = PREDICT_MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".png"
    if not os.path.exists(mask_path):
        print("{} has no predict mask!".format(imgname))
    mask = skimage.io.imread(mask_path).astype(np.float32)
    mask = skimage.color.rgb2grey(mask)
    mask = np.where(mask >= 127.5, 1, 0).astype(np.float32)

    return mask

# =====================================
# Calculate precision, recall, and MAE.
# =====================================
def cal_precision_recall_mae(prediction, gt):
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction >= threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae

# ====================
# Calculate f measure.
# ====================
def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure

# =============
# Main function
# ==============
def main():
    results = OrderedDict()

    gt_path = f'{testing_path}/{dataset_name}/mask'
    prediction_path =  f'{result_path}/{dataset_name}'

    precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    mae_record = AvgMeter()

    img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_path) if f.endswith('.png')]

    # Iterate through the generated and ground-truth masks.
    for idx, img_name in enumerate(img_list):
        print('evaluating for %s: %d / %d      %s' % (dataset_name, idx + 1, len(img_list), img_name + '.png'))

        prediction = np.array(Image.open(os.path.join(prediction_path, img_name + '.png')).convert('L'))
        gt = np.array(Image.open(os.path.join(gt_path, img_name + '.png')).convert('L'))

        precision, recall, mae = cal_precision_recall_mae(prediction, gt)
        for idx, data in enumerate(zip(precision, recall)):
            p, r = data
            precision_record[idx].update(p)
            recall_record[idx].update(r)

        # Calculate the minmum average error.
        mae_record.update(mae)

    # Calculate the F_beta score.
    fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                            [rrecord.avg for rrecord in recall_record])

    results[dataset_name] = OrderedDict([('F', "%.4f" % fmeasure), ('mae', "%.4f" % mae_record.avg)])

    print(results[dataset_name])

if __name__ == '__main__':
    main()

