from __future__ import print_function
from show3d_balls import showpoints
import argparse
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetDenseCls
from tqdm import tqdm
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='../pretrained_networks/latest_segmentation.pt', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0', help='dataset path')
parser.add_argument('--class_choice', type=str, default='Chair', help='class choice')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(num_classes= state_dict['model']['layer4.weight'].size()[0], feature_transform=opt.feature_transform)
classifier.load_state_dict(state_dict['model'])
classifier.cuda()
classifier.eval()

shape_ious = []
with torch.no_grad():
	for i,data in tqdm(enumerate(dataloader, 0)):
		points, target = data
		points = points.transpose(2, 1)
		points, target = points.cuda(), target.cuda()
		pred, _, _ = classifier(points)
		pred_choice = pred.data.max(2)[1]
		pred_np = pred_choice.cpu().data.numpy()
		target_np = target.cpu().data.numpy() - 1
	
		for shape_idx in range(target_np.shape[0]):
			parts = range(dataset.num_seg_classes)#np.unique(target_np[shape_idx])
			part_ious = []
			for part in parts:
				I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
				U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
				if U == 0:
					iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
				else:
					iou = I / float(U)
				part_ious.append(iou)
			shape_ious.append(np.mean(part_ious))
	print("mIOU for class {}: {:.4f}".format(opt.class_choice, np.mean(shape_ious)))

idx = opt.idx

print("model %d/%d" % (idx+1, len(dataset)))
point, seg = dataset[idx]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("gist_rainbow", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

point=point.cuda()
point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point)
pred_choice = pred.cpu().data.max(dim=2)[1]
print(pred_choice)

pred_color = cmap[pred_choice.numpy()[0], :]

showpoints(point_np, gt, pred_color)
