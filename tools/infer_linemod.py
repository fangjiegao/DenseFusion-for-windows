import _init_paths
import argparse
import copy
import time

import cv2
import numpy as np
import open3d as o3d
import torch.utils.data
from torch.autograd import Variable

from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix


def show_3D(image, K, target, color, name):
    d2_list = []
    for i in range(target.shape[0]):
        d2 = np.transpose(np.matmul(K, np.transpose(target[i])))
        d2 = d2 / d2[2]
        d2_list.append(d2[:2])
    d3 = np.array(d2_list, dtype=np.int32)
    for i in range(d3.shape[0]):
        image = cv2.circle(image, center=(d3[i][0], d3[i][1]), radius=1, color=color)

    cv2.imshow('image_window', image)
    # Wait for any key to close the window
    cv2.waitKey(0)
    cv2.imwrite(opt.out_root + '/visual/{0}.jpg'.format(name), image)


# 1.创建解析器
parser = argparse.ArgumentParser()

# 2.添加参数
# 验证数据根目录
parser.add_argument('--dataset_root', type=str, default='D:\Linemod_preprocessed', help='dataset root dir')
# 推理结果保存目录
parser.add_argument('--out_root', type=str, default='F:\\pythonProject\\DenseFusion-master\\infer_linemod_out',
                    help='out root dir')
# 主干网络的模型，也就是PoseNet网络模型
parser.add_argument('--model', type=str,
                    # default='D:\\trained_checkpoints\\trained_checkpoints\linemod\\no_refine_pose_model_15_0.011377632944034037.pth',
                    default='F:\\pythonProject\\DenseFusion-master\\trained_models\\linemod\\pose_model_current.pth',
                    help='resume PoseNet model')
# 姿态提炼网络模型，及PoseRefineNet网络模型
parser.add_argument('--refine_model', type=str,
                    # default='D:\\trained_checkpoints\\trained_checkpoints\linemod\\pose_refine_model_493_0.006761023565178073.pth',
                    default='F:\\pythonProject\\DenseFusion-master\\trained_models\\linemod\\pose_refine_model_current.pth',
                    help='resume PoseRefineNet model')
opt = parser.parse_args()
# 3.解析参数
opt = parser.parse_args()

# 测试目标物体的总数目
num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
color_list = [(80, 196, 143), (38, 204, 216), (54, 133, 254), (153, 119, 239),
              (245, 97, 111), (247, 177, 63), (249, 226, 100), (244, 122, 117),
              (0, 157, 178), (2, 75, 81), (7, 128, 207), (118, 80, 5), (0, 0, 0)]

num_points = 1000  # 根据当前需要测试帧的RGB-D，生成的点云数据，随机选择其中500个点云。
iteration = 4
bs = 1

# 相机矩阵
cam_cx = 325.26110
cam_cy = 242.04899
cam_fx = 572.41140
cam_fy = 573.57043
K = np.array([[cam_fx, 0.0, cam_cx],
              [0.0, cam_fy, cam_cy],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# PoseNet网络模型得类构建，此时还没有进行前向传播
estimator = PoseNet(num_points=num_points, num_obj=num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
refiner.cuda()

# PoseNet模型参数加载-加载模型
estimator.load_state_dict(torch.load(opt.model))
# PoseRefineNet模型参数加载-加载模型
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

# 以评估得模式加载数据
testdataset = PoseDataset_linemod('test', num_points, False, opt.dataset_root, 0.0, True)
# 转化为torch迭代器形式
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)

item_count = 0
ori_img_path = []
for obj in objlist:
    input_file = open('{0}/data/{1}/test.txt'.format(opt.dataset_root, '%02d' % obj))
    while 1:
        # 记录处理的数目
        item_count += 1
        input_line = input_file.readline()  # 0000
        # test模式下，图片序列为10的倍数则continue
        if item_count % 10 != 0:
            continue
        # 文件读取完成
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        # 把RGB图像的路径加载到self.list_rgb列表中
        ori_img_path.append('{0}/data/{1}/rgb/{2}.png'.format(opt.dataset_root, '%02d' % obj, input_line))

for i, data in enumerate(testdataloader):
    # 先获得一个样本需要的数据
    # points, choose, img, target, model_points, idx, mask = data
    points, choose, img, target, model_points, idx = data
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        continue

    #points, choose, img, target, model_points, idx, mask = Variable(points).cuda(), \
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
        Variable(choose).cuda(), \
        Variable(img).cuda(), \
        Variable(target).cuda(), \
        Variable(model_points).cuda(), \
        Variable(idx).cuda()  # ,Variable(mask).cuda()

    inner_time = time.time()
    # 通过PoseNet，预测当前帧的poses
    # pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx, mask)
    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    # 从所有的poses中找到最好的那一个，及置信度最高的那个
    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    # 进行PoseRefineNet网络的循环迭代
    for ite in range(0, iteration):
        # 前面得到的结果已经不是torch格式的数据了，所以这里进行一个转换
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points,
                                                                                         1).contiguous().view(1,
                                                                                                              num_points,
                                                                                                              3)

        # 把姿态旋转参数r转换为矩阵形式
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        # 把points进行一个逆操作得到new_points
        new_points = torch.bmm((points - T), R).contiguous()
        # 进行提炼，得到新的姿态pose
        pred_r, pred_t = refiner(new_points, emb, idx)

        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        # 转化为四元数的矩阵
        my_mat_2 = quaternion_matrix(my_r_2)
        # 获得偏移参数（矩阵）
        my_mat_2[0:3, 3] = my_t_2

        # my_mat第一次迭代是主干网路初始的预测结果，之后是上一次迭代之后最后输出的结果
        # my_mat_2是当前迭代refiner预测出来的结果，矩阵相乘之后，把其当作该次迭代最后的结果
        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0

        # 同样转化为四元数矩阵
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        # 把当前最后的估算结果，赋值给送入下次迭代的pose
        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final  # 'my_r': quaternion
        my_t = my_t_final  # 'my_t': translation

    model_points = model_points[0].cpu().detach().numpy()
    # 转化为四元数矩阵，取出其中的旋转矩阵，这里的my_r是最后预测的结果
    my_r = quaternion_matrix(my_r)[:3, :3]

    # model_points经过姿态转化之后的点云数据
    pred = np.dot(model_points, my_r.T) + my_t

    # 获得测试图片
    ori_img = cv2.imread(ori_img_path[i])
    # 保存点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred)
    o3d.io.write_point_cloud(opt.out_root + '/ply/{0}.ply'.format(i), pcd)
    # 可视化
    print("pred:", pred.shape)
    print("K:", K)
    # show_3D(ori_img, K, target=pred, color=color_list[idx.item()], name='{}'.format(i))
