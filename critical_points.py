from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F
from show3d_balls import showpoints
import matplotlib.pyplot as plt
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--critical_point',action='store_true',help='On: display critical points; OFF: display all points')
parser.add_argument('--idx',type=int,default=0,help='index of the displayed instance')
parser.add_argument('--class_choice',type=str,default='Chair',help='type of the displayed instance')
parser.add_argument('--feature_transform',action='store_true')

opt = parser.parse_args()
print(opt)

dataset=ShapeNetDataset(root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',class_choice=opt.class_choice,classification=True,split='test',data_augmentation=False)

classifier = PointNetCls(num_classes=len(dataset.seg_classes),feature_transform=opt.feature_transform)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model)['model'])
classifier.eval()

div=7
cyclic=0
tol=1e-6
axis=0 # axis = 0 ~ 2
point,_=dataset[opt.idx]
original_point_np=point.numpy()
#cmap options:
#for "gist_rainbow":
#set div=7, cyclic=0
#for "rainbow":
#set div=7, cyclic=0
#for "hsv":
#set div=8, cyclic=1
cmap = plt.cm.get_cmap("gist_rainbow", div)
cmap = np.array([cmap(i) for i in range(div-cyclic)])[:, :3]
maximum=np.max(original_point_np[:,axis])
minimum=np.min(original_point_np[:,axis])
choice=np.floor((original_point_np[:,axis]-minimum)/(maximum-minimum+tol)*float(div-cyclic)).astype(int)
color_original=cmap[choice,:]

point=point.unsqueeze(0).transpose(1,2).cuda()
_,_,_,idx=classifier(point)
idx=idx.squeeze(0).squeeze(-1)
critical_point=torch.index_select(point,2,idx).squeeze(0).transpose(0,1)
critical_point_np=critical_point.cpu().numpy()

maximum=np.max(critical_point_np[:,axis])
minimum=np.min(critical_point_np[:,axis])
choice=np.floor((critical_point_np[:,axis]-minimum)/(maximum-minimum+tol)*float(div-cyclic)).astype(int)
color_critical=cmap[choice,:]

if opt.critical_point:
	showpoints(critical_point_np, color_critical)
else:
	showpoints(original_point_np, color_original)
