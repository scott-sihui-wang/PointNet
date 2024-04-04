from __future__ import print_function

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.k = k

        # Each layer has batchnorm and relu on it
        #TODO
        # layer 1: k -> 64

        self.layer1=nn.Sequential(
            nn.Conv1d(self.k,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )

        #TODO
        # layer 2:  64 -> 128

        self.layer2=nn.Sequential(
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )

        #TODO
        # layer 3: 128 -> 1024

        self.layer3=nn.Sequential(
            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
            )
        
        #TODO
        # fc 1024 -> 512

        self.fc_layer1=nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
            )

        #TODO
        # fc 512 -> 256

        self.fc_layer2=nn.Sequential(
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU()
            )

        #TODO
        # fc 256 -> k*k (no batchnorm, no relu)

        self.fc_layer3=nn.Linear(256,self.k*self.k)

        #TODO
        # ReLU activationfunction

        self.relu=nn.ReLU()
        


    def forward(self, x):
        batch_size, _, num_points = x.shape
        #TODO
        # apply layer 1

        x=self.layer1(x)

        #TODO
        # apply layer 2

        x=self.layer2(x)

        #TODO
        # apply layer 3

        x=self.layer3(x)

        #TODO
        # do maxpooling and flatten

        pool=nn.MaxPool1d(num_points)
        x=pool(x)
        x=x.view(-1,1024)

        #TODO
        # apply fc layer 1

        x=self.fc_layer1(x)

        #TODO
        # apply fc layer 2

        x=self.fc_layer2(x)

        #TODO
        # apply fc layer 3

        x=self.fc_layer3(x)

        #TODO
        #reshape output to a b*k*k tensor

        x=x.view(-1,self.k,self.k)

        #TODO
        # define an identity matrix to add to the output. This will help with the stability of the results since we want our transformations to be close to identity

        Id=Variable(torch.eye(self.k).unsqueeze(0).repeat(batch_size,1,1))
        if x.is_cuda:
            Id=Id.cuda()
        x=x+Id

        #TODO
        # return output
        return x



class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = True):
        super(PointNetfeat, self).__init__()

        self.feature_transform= feature_transform
        self.global_feat = global_feat

        #TODO
        # Use TNet to apply transformation on input and multiply the input points with the transformation

        self.trans1=TNet(3)

        #TODO
        # layer 1:3 -> 64

        self.layer1=nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )

        #TODO
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)

        self.trans2=TNet(64)

        #TODO
        # layer2: 64 -> 128

        self.layer2=nn.Sequential(
            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )

        #TODO
        # layer 3: 128 -> 1024 (no relu)

        self.layer3=nn.Sequential(
            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024)
            )

        #TODO
        # ReLU activation

        self.relu=nn.ReLU()

    def forward(self, x):
        batch_size, _, num_points = x.shape
        #TODO
        # input transformation, you will need to return the transformation matrix as you will need it for the regularization loss

        trans_pts=self.trans1(x).transpose(2,1)
        x=torch.bmm(trans_pts,x)

        #TODO
        # apply layer 1
        x=self.layer1(x)

        #TODO
        # feature transformation, you will need to return the transformation matrix as you will need it for the regularization loss

        if self.feature_transform:
            trans_feat=self.trans2(x).transpose(2,1)
            x=torch.bmm(trans_feat,x)

        feat=x

        #TODO
        # apply layer 2

        x=self.layer2(x)

        #TODO
        # apply layer 3

        x=self.layer3(x)

        #TODO
        # apply maxpooling

        #pool=nn.MaxPool1d(num_points)
        #x=pool(x)
        x,idx=torch.max(x,dim=2,keepdim=True)
        x=x.view(-1,1024)


        #TODO
        # return output, input transformation matrix, feature transformation matrix
        if self.global_feat: # This shows if we're doing classification or segmentation
            if self.feature_transform:
                return x,trans_pts,trans_feat,idx
            else:
                return x,trans_pts,None,idx

        else:
            x=x.unsqueeze(2)
            x=x.repeat(1,1,num_points)
            x=torch.cat([x,feat],dim=1)
            if self.feature_transform:
                return x,trans_pts,trans_feat,None
            else:
                return x,trans_pts,None,None


class PointNetCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans_pts, trans_feat, idx = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans_pts, trans_feat, idx


class PointNetDenseCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        #TODO
        # get global features + point features from PointNetfeat

        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.num_classes=num_classes

        #TODO
        # layer 1: 1088 -> 512

        self.layer1=nn.Sequential(
            nn.Conv1d(1088,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU()
            )

        #TODO
        # layer 2: 512 -> 256

        self.layer2=nn.Sequential(
            nn.Conv1d(512,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU()
            )

        #TODO
        # layer 3: 256 -> 128

        self.layer3=nn.Sequential(
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )

        #TODO
        # layer 4:  128 -> k (no ru and batch norm)

        self.layer4=nn.Conv1d(128,self.num_classes,1)

        #TODO
        # ReLU activation

        self.relu=nn.ReLU()
    
    def forward(self, x):
        #TODO
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)

        b,_,n=x.shape
        x,trans_pts,trans_feat,_=self.feat(x)

        #TODO
        # apply layer 1

        x=self.layer1(x)

        #TODO
        # apply layer 2

        x=self.layer2(x)

        #TODO
        # apply layer 3

        x=self.layer3(x)

        #TODO
        # apply layer 4

        x=self.layer4(x)

        #TODO
        # apply log-softmax

        x=x.transpose(2,1).contiguous()
        x=x.view(-1,self.num_classes)
        x=F.log_softmax(x,dim=-1)
        x=x.view(b,n,self.num_classes)
        return x,trans_pts,trans_feat

        


def feature_transform_regularizer(trans):

    batch_size, feature_size, _ = trans.shape
    #TODO
    # compute I - AA^t

    I=torch.eye(feature_size)
    if trans.is_cuda:
        I=I.cuda()
    mat=I-torch.bmm(trans,trans.transpose(2,1))

    #TODO
    # compute norm

    fro_norm=torch.linalg.norm(mat,ord='fro',dim=(1,2))

    #TODO
    # compute mean norms and return
    loss=torch.mean(fro_norm)
    return loss



if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_classes = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_classes = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())

