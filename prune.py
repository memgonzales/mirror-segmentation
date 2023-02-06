# ===========================================================================================================
# THIS FILE CONTAINS THE SCRIPT FOR MODEL PRUNING AND GENERATING THE PREDICTED MASKS USING THE PRUNED MODEL.
# Authors: Mark Edward M. Gonzales & Lorene C. Uy
# ===========================================================================================================

import nni
from nni.compression.pytorch.pruning import *
from nni.compression.pytorch import apply_compression_results
from nni.compression.pytorch.utils.counter import count_flops_params

import torch

from config import testing_path, pruned_weights_path, weights_path, dataset_name, result_path
from pmd import PMDLite

# Change this to the device ordinal of the GPU
# If the device is cuda:x, device_ids should be [x].
device_ids = [0]

# Create a dummy tensor for counting the number of FLOPS.
dummy = torch.rand(1, 3, 416, 416).cuda(device_ids[0])

# Adjust the sparsity level as needed.
sparsity = 0.1

# Perform pruning on both convolutional and linear layers.
config_list = [{'sparsity_per_layer': sparsity, 
                'op_types': ['Conv2d', 'Linear']}]

# Load model weights and biases. Change the device ordinal as needed.
model = PMDLite().cuda(device_ids[0])
model.load_state_dict(torch.load(weights_path, map_location='cuda:0'))

# Perform filer pruning via geometric median.
pruner = FPGMPruner(model, config_list)
_, masks = pruner.compress()

# ========================================================================================================
# To retrain the model, append the contents of train.py, but change the model to the pruned model (model)
# instead of PMDLite.

# In our study, we performed retraining for 20 epochs:
# - Learning rate rewinding, which uses the original learning rate schedule to retrain
#   unpruned weights from their final values, was adopted.
# - Learning rate rewinding was proposed in:
#    A. Renda, J. Frankle, and M. Carbin, "Comparing rewinding and finetuning in neural network pruning,"
#        in International Conference on Learning Representations, 2020.
# ========================================================================================================

# =========================================================================
# To perform prediction using the mode, append the contents of predict.py,
# but change the model to the pruned model (model) instead of PMDLite,
# as shown below.
# =========================================================================

import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from misc import check_mkdir, crf_refine

# Change this to the device ordinal of the GPU
# If the device is cuda:x, device_ids should be [x].
device_ids = [0]
torch.cuda.set_device(device_ids[0])

# Use a fully connected conditional random field for post-processing.
# Proposed in:
#    P. Krahenb ¨ uhl and V. Koltun, “Efficient inference in fully connected ¨
#        CRFs with Gaussian edge potentials,” in Advances in Neural Information
#        Processing Systems, J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira,
#        and K. Weinberger, Eds., vol. 24. Curran Associates, Inc., 2011.
args = {
    'scale': 384,
    'crf': True
}

# ====================================
# Apply transformation to the images.
# ====================================
img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

# Set path to test dataset.
to_test = {dataset_name: testing_path}

# ==============
# Main function
# ==============
def main():
    net = model.cuda(device_ids[0])

    # Load model weights and biases. Change the device ordinal as needed.
    net.load_state_dict(torch.load(pruned_weights_path, map_location='cuda:0'))

    DS = dataset_name
    
    # Start the generation of the predicted mirror masks.
    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, DS, 'image'))]

            start = time.time()

            # Iterate through every test image.
            for idx, img_name in enumerate(img_list):
                print('Predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                check_mkdir(os.path.join(result_path, name))

                img = Image.open(os.path.join(root, DS, 'image/', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()

                # Transfer the outputs (sigmoid of the mirror and edge maps) of the network 
                # to the CPU.
                f_4, f_3, f_2, f_1, edge, final = net(img_var)
                f_4 = f_4.data.squeeze(0).cpu()
                f_3 = f_3.data.squeeze(0).cpu()
                f_2 = f_2.data.squeeze(0).cpu()
                f_1 = f_1.data.squeeze(0).cpu()
                edge = edge.data.squeeze(0).cpu()
                final = final.data.squeeze(0).cpu()

                f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                edge = np.array(transforms.Resize((h, w))(to_pil(edge)))
                final = np.array(transforms.Resize((h, w))(to_pil(final)))

                # Perform post-processing using a fully connected conditional random field.
                if args['crf']:
                    final = crf_refine(np.array(img.convert('RGB')), final)

                # Save the mask to the results folder.
                Image.fromarray(final).save(os.path.join(result_path, name, img_name[:-4] + ".png"))

            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))

if __name__ == '__main__':
    main()
