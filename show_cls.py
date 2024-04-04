from __future__ import print_function
import argparse
import random
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    split='train',
    npoints=2500,
    data_augmentation=False)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)
val_dataset = ShapeNetDataset(
    root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    split='val',
    npoints=2500,
    data_augmentation=False)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)
test_dataset = ShapeNetDataset(
    root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    split='test',
    npoints=2500,
    data_augmentation=False)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

classifier = PointNetCls(num_classes=len(test_dataset.classes),feature_transform=opt.feature_transform)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model)['model'])
classifier.eval()


total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        #TODO
        # calculate average classification accuracy

        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _, _ = classifier(points)
        pred_labels = torch.max(preds, dim= 1)[1]

        total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
        total_targets = np.concatenate([total_targets, target.cpu().numpy()])

    correct=(total_targets==total_preds)
    accuracy = 100 * correct.sum() / len(test_dataset)
    print('Test Accuracy (Overall) = {}/{} = {:.2f}%'.format(correct.sum(),len(test_dataset),accuracy))
    key_list=list(test_dataset.classes.keys())
    for i in range(len(test_dataset.classes)):
    	cls=(total_targets==i)
    	correct_cls=np.array([correct[idx] and cls[idx] for idx in range(len(correct))])
    	accuracy_cls=100*correct_cls.sum()/cls.sum()
    	print('Test Accuracy (class {}) = {}/{} = {:.2f}%'.format(key_list[i],correct_cls.sum(),cls.sum(),accuracy_cls))
print('')    	
total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        #TODO
        # calculate average classification accuracy

        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _, _ = classifier(points)
        pred_labels = torch.max(preds, dim= 1)[1]

        total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
        total_targets = np.concatenate([total_targets, target.cpu().numpy()])

    correct=(total_targets==total_preds)
    accuracy = 100 * correct.sum() / len(dataset)
    print('Train Accuracy (Overall) = {}/{} = {:.2f}%'.format(correct.sum(),len(dataset),accuracy))
    key_list=list(test_dataset.classes.keys())
    for i in range(len(test_dataset.classes)):
    	cls=(total_targets==i)
    	correct_cls=np.array([correct[idx] and cls[idx] for idx in range(len(correct))])
    	accuracy_cls=100*correct_cls.sum()/cls.sum()
    	print('Train Accuracy (class {}) = {}/{} = {:.2f}%'.format(key_list[i],correct_cls.sum(),cls.sum(),accuracy_cls))
print('')    	
total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(val_dataloader, 0):
        #TODO
        # calculate average classification accuracy

        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _, _ = classifier(points)
        pred_labels = torch.max(preds, dim= 1)[1]

        total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
        total_targets = np.concatenate([total_targets, target.cpu().numpy()])

    correct=(total_targets==total_preds)
    accuracy = 100 * correct.sum() / len(val_dataset)
    print('Validation Accuracy (Overall) = {}/{} = {:.2f}%'.format(correct.sum(),len(val_dataset),accuracy))
    key_list=list(test_dataset.classes.keys())
    for i in range(len(test_dataset.classes)):
    	cls=(total_targets==i)
    	correct_cls=np.array([correct[idx] and cls[idx] for idx in range(len(correct))])
    	accuracy_cls=100*correct_cls.sum()/cls.sum()
    	print('Validation Accuracy (class {}) = {}/{} = {:.2f}%'.format(key_list[i],correct_cls.sum(),cls.sum(),accuracy_cls))

