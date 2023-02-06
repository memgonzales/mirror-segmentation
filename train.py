# ======================================================
# THIS FILE CONTAINS THE METHOD FOR TRAINING THE MODEL.
# ======================================================

# Reference source code:
#    H. Mei (2021). "Camouflaged object segmentation with distraction mining,"
#        2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 
#        pp. 8768-8877, 2021.
#    Repository: https://github.com/Mhaiyang/CVPR2021_PFNet

# Mark Edward M. Gonzales & Lorene C. Uy:
# - Added annotations and comments
# - Added code for saving and loading the optimizer's state (aside from the model's state)
# - Changed the loss function to our work's proposed compound loss function

import datetime
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import training_path, checkpoint_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir

# Change "pmd" to the appropriate version when running experiments with other models.
# By default, pmd refers to our best-performing unpruned model.
from pmd import PMDLite

import loss
import gc

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

cudnn.benchmark = True

torch.manual_seed(2021)

# Change this to the device ordinal of the GPU
# If the device is cuda:x, device_ids should be [x].
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_ids = [0]

# Change this to the path of the directory containing the epoch snapshots.
ckpt_path = checkpoint_path
exp_name = 'PMDLite'

# To load from epoch snapshot x, change 'last_epoch' to x and 'snapshot' to 'x'.
# Use stochastic gradient descent for minimizing the loss function
#    and polynomial decay for updating the learning rate.
args = {
    'epoch_num': 150,
    'train_batch_size': 10,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 352,
    'save_point': [i for i in range(1, 151)],
    'poly_train': True,
    'optimizer': 'SGD',
}

# Change this to the path of the directory containing the epoch snapshots.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))

# Change this to the path of the directory containing the logs of each epoch.
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# ====================================
# Apply transformation to the images.
# ====================================
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])

img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # ImageNet mean and std
])

edge_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()

# ===========================
# Load the training dataset.
# ===========================
train_set = ImageFolder(training_path, joint_transform, img_transform, edge_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

# ==============================
# Initialize the loss functions.
# ===============================
structure_loss = loss.structure_loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = loss.IOU().cuda(device_ids[0])
edge_loss = loss.edge_loss().cuda(device_ids[0])

# ==============
# Main function
# ==============
def main():
    # Initialize the model.
    net = PMDLite(training = True).cuda(device_ids[0])

    # Initialize the optimizer.
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
            'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
            'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    # Resume training from epoch snapshot.
    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'O' + args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

# =============================
# Method for trainng the model
# =============================
def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()

    # Iterate per epoch.
    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record, loss_5_record, loss_6_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))

        # Iterate per batch.
        for data in train_iterator:
            # Update learning rate via polynomial strategy.
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, edges, labels = data
            batch_size = inputs.size(0)

            inputs = inputs.to(device)
            edges = edges.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Feed data to the model.
            layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_edge, final_predict = net(inputs)

            # Loss for the multi-scale mirror maps
            loss_1 = iou_loss(layer4_predict, labels)
            loss_2 = iou_loss(layer3_predict, labels)
            loss_3 = iou_loss(layer2_predict, labels)
            loss_4 = iou_loss(layer1_predict, labels)
            # Loss for the boundary maps
            loss_5 = edge_loss(layer0_edge, edges)
            # Loss for the final (output) mirror map
            loss_6 = structure_loss(final_predict, labels)

            # The weighting coefficients were determined epirically.
            loss = 1 * loss_1 + 1 * loss_2 + 1 * loss_3 + 1 * loss_4 + 5 * loss_5 + 2 * loss_6

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Record loss values to a log file.
            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_5_record.update(loss_5.data, batch_size)
            loss_6_record.update(loss_6.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_5', loss_4, curr_iter)
                writer.add_scalar('loss_6', loss_4, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg,
                   loss_3_record.avg, loss_4_record.avg, loss_5_record.avg, loss_6_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')
            print("Log written --")

            curr_iter += 1

            # Collect garbage to prevent CUDA out-of-memory error.
            del inputs
            del labels
            gc.collect()
            torch.cuda.empty_cache()

        # Save epoch snapshot.
        if epoch in args['save_point']:
            # Transfer to CPU.
            net.cpu()
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name + '/%d.pth' % epoch))
            torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name + '/O%d.pth' % epoch))
            # Transfer back to GPU.
            net.cuda(device_ids[0])
            print("Epoch snapshot saved!")

        # Finish training.
        if epoch >= args['epoch_num']:
            # Transfer to CPU.
            net.cpu()
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '/%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization finished!")
            return

if __name__ == '__main__':
    main()