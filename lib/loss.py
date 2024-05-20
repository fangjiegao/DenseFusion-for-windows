from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
#from lib.knn.__init__ import KNearestNeighbor
from sklearn.neighbors import NearestNeighbors


def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
    # knn = KNearestNeighbor(1)
    knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    bs, num_p, _ = pred_c.size() # (1,500,1)

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))  # 四元数 (1,500,4)
    # 旋转矩阵(500,3,3)
    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()  # #转置/逆(500,3,3)
    # #把model_points复制500份, model_points是加载的点云文件
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3) # (1,500,3) -> (500,500,3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)  # (1,500,3)  -> (500,1,3)
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)  # (1,500,3)  -> (500,1,3)
    pred_c = pred_c.contiguous().view(bs * num_p)  # 置信度 (1,500, 1)  -> (500)
    # 将model_points旋转矩阵的逆后。points：crop后生成的点云
    pred = torch.add(torch.bmm(model_points, base), points + pred_t)  # bmm -> batch matrix multiplication (500, 500, 3)
    if not refine:
        if idx[0].item() in sym_list:  # 对称数据
            """
            target = target[0].transpose(1, 0).contiguous().view(3, -1)  # (500,500,3) -> (3, 500)
            pred = pred.permute(2, 0, 1).contiguous().view(3, -1)  # (500,500,3) -> (3, 250000)
            t = target.unsqueeze(0)  # (1, 3, 500)
            p = pred.unsqueeze(0)  # (1, 3, 250000)
            inds = torch.tensor(list(range(500)))  #
            print(inds.view(-1).detach() - 1)
            # target = torch.index_select(target, 1, inds.view(-1).detach() - 1)
            target = torch.index_select(target, 1, inds.cuda())  # 找出inds对应的点(x,y,z) (3, len(inds))
            pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()  # (3, 250000) ->(500, 500, 3)
            target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous() # (3, 250000) ->(500, 500, 3)
            """
            target = target[0].transpose(1, 0).contiguous().view(3, -1)  # (500,500,3) -> (3, 500)
            pred = pred.permute(2, 0, 1).contiguous().view(3, -1)  # (500,500,3) -> (3, 250000)
            #inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
            target_numpy = target.cpu().detach().numpy().transpose(1, 0)
            pred_numpy = pred.cpu().detach().numpy().transpose(1, 0)
            knn.fit(target_numpy)
            distances, inds = knn.kneighbors(pred_numpy)
            inds = inds.flatten()
            distances_tensor = torch.tensor(distances).cuda()
            distances_tensor = distances_tensor.view(1, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            # target = torch.index_select(target, 1, inds.view(-1) - 1)
            # target = torch.index_select(torch.tensor(target_numpy).cuda(), 0, torch.tensor(inds).cuda())
            # target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            # pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            # dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
            dis = torch.mean(distances_tensor, dim=1).squeeze()
            loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)
        else:
            dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
            loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)
    else:
        dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
        loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)

    # dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)  # |pred - target|^2 方差
    # loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)

    # dis = dis * pred_c
    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    dis = dis.view(bs, num_p)


    t = ori_t[which_max[0]] + points[which_max[0]]
    points = points.view(1, bs * num_p, 3)

    ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3)
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    # del knn

    if idx[0].item() in sym_list:
        # print("对称：", idx, loss, dis[0][which_max[0]], pred.shape, target.shape)
        pass
    else:
        # print("非对称：", idx, loss, dis[0][which_max[0]], pred.shape, target.shape)
        pass
    return loss, dis[0][which_max[0]], new_points.detach(), new_target.detach()


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)
