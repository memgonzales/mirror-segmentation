# ==================================================================
# THIS FILE CONTAINS THE METHOD FOR GENERATING THE PREDICTED MASKS.
# ==================================================================

# Reference source code:
#    J. Lin, G. Wang, and R. H. Lau, "Progressive mirror detection,” in 2020
#        IEEE/CVF Conference on Computer Vision and Pattern Recognition
#        (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, June 2020,
#        pp. 3694–3702.
#    Repository: https://jiaying.link/cvpr2020-pgd/

# Mark Edward M. Gonzales & Lorene C. Uy:
# - Added annotations and comments

import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from config import testing_path, weights_path, dataset_name, result_path
from misc import check_mkdir, crf_refine


# Change "pmd" to the appropriate version when running experiments with other models.
# By default, pmd refers to our best-performing unpruned model.
from pmd import PMDLite

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
    net = PMDLite().cuda(device_ids[0])

    # Load model weights and biases. Change the device ordinal as needed.
    net.load_state_dict(torch.load(weights_path, map_location='cuda:0'))

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
