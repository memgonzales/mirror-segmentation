# =======================================================
# THIS FILE CONTAINS THE METHODS FOR DATASET PROCESSING.
# =======================================================

# Reference source code:
#    J. Lin, G. Wang, and R. H. Lau, "Progressive mirror detection,” in 2020
#        IEEE/CVF Conference on Computer Vision and Pattern Recognition
#        (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, June 2020,
#        pp. 3694–3702.
#    Repository: https://jiaying.link/cvpr2020-pgd/

# Mark Edward M. Gonzales & Lorene C. Uy:
# - Added annotations and comments

import os
import os.path
import torch.utils.data as data
from PIL import Image

# ======================================
# Create the directory for the datasets.
# ======================================
def make_dataset(root):
    image_path = os.path.join(root, 'image')
    mask_path = os.path.join(root, 'mask')
    edge_path = os.path.join(root, 'edge')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), 
        os.path.join(edge_path, img_name + '.png'),
        os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

# =============================
# Class for creating a dataset.
# =============================
class ImageFolder(data.Dataset):
    
    # Instantiate a dataset object
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, edge_transform = None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform

    # Return a sample from a dataset
    def __getitem__(self, index):
        img_path, edge_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        edge = Image.open(edge_path).convert('L')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, edge, target = self.joint_transform(img, edge, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.edge_transform is not None:
            edge = self.edge_transform(edge)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, edge, target

    # Return number of samples in dataset
    def __len__(self):
        return len(self.imgs)