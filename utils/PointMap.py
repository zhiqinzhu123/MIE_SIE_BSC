"""
PointMap.py 用于生成边界点图
    Inputs： xxx_seg.nii.gz
    Returns: xxx_pm.seg.nii.gz
"""

from skimage.feature import canny
from scipy.interpolate import griddata
from tqdm import tqdm
from glob import glob
import numpy as np
import SimpleITK as sitk
import os
import random

# canny算子函数取图像边界面
def canny_3d(mask_paths):
    Mask = mask_paths
    dim = np.shape(Mask)

    MIN_CANNY_THRESHOLD = 0.5
    MAX_CANNY_THRESHOLD = 0.5
    
    edges_x = np.zeros(dim, dtype=bool) 
    edges_y = np.zeros(dim, dtype=bool) 
    edges_z = np.zeros(dim, dtype=bool) 
    edges = np.zeros(dim, dtype=bool) 

    for l in range(dim[3]):    
        for i in range(dim[2]):
            edges_x[:,:,i,l] = canny(Mask[:,:,i,l], low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD, sigma = 0)        
        for j in range(dim[1]):
            edges_y[:,j,:,l] = canny(Mask[:,j,:,l], low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD, sigma = 0)            
        for k in range(dim[0]):
            edges_z[k,:,:,l] = canny(Mask[k,:,:,l], low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD, sigma = 0)
                
    # edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD)
        for i in range(dim[2]):
            for j in range(dim[1]):
                for k in range(dim[0]):
                    edges[k,j,i,l] = (edges_x[k,j,i,l] and edges_y[k,j,i,l]) or (edges_x[k,j,i,l] and edges_z[k,j,i,l]) or (edges_y[k,j,i,l] and edges_z[k,j,i,l])
                        #edges[i,j,k] = (edges_x[i,j,k]) or (edges_y[i,j,k]) or (edges_z[i,j,k])
    return edges

# 随机选择n个边界点
def point_select(bound, data, num):
    dim = np.shape(bound)
    # 建一个空白矩阵准备索引
    select_points = np.zeros([155,240,240,3])
    # 建一个list装每个点的坐标
    select_point_index = []
    # 建一个list装每个点的值    
    # select_point_value = []

    for j in range(dim[3]) :
        n = num[j]
        point = np.argwhere(bound[:,:,:,j] != 0)
        point.tolist()
        if len(point) < n :
            n = len(point)
        rand_point_list = random.sample(list(point), n)
        rand_point_index = np.array(rand_point_list)

        for i in rand_point_index:
            ix = i[2]
            iy = i[1]
            iz = i[0]
            select_points[iz, iy, ix, j] = data[iz, iy, ix, j]
            select_point_index.append([iz, iy, ix, j])
            # select_point_value.append(select_points[j, iz, iy, ix])

    return select_points, select_point_index #, select_point_value

# 创建插值包围区域
def Griddata(index):
    # dx, pts = 159, 160j
    Z,Y,X = np.mgrid[0:154:155j, 0:239:240j, 0:239:240j]
    pointmap = np.zeros([155,240,240,3]) 

    for i in range(3):
        R = index[index[:,3] == i]
        if R.shape[0] > 0:
            R = R[:, :3]
            V = np.ones(len(R))
            F = griddata(R, V, (Z,Y,X), method='linear')
            # contour3d(F,contours=8,opacity=.2 )
            # show()
            pointmap[:,:,:,i] = F[:, :, :]
    return pointmap

# 得到best_points在每个分区的数量
def get_num(index):
    select_num = np.zeros(3)
    for c in range(3):
        select_index = index[index[:,3] == c]
        select_num[c] = select_index.shape[0]
    return select_num

# 把空间点半径领域为r的区域置1
def point_circle(index, r):
    ONE = np.zeros([155,240,240,3])
    for c in range(3):
        p_index = index[index[:,3] == c]
        num = p_index.shape[0]
        for i in range(num):
            center_x = p_index[i, 2]
            center_y = p_index[i, 1]
            center_z = p_index[i, 0] 
            for ix in range(center_x - r, center_x + r + 1):
                for iy in range(center_y - r, center_y + r + 1):
                    for iz in range(center_z - r, center_z + r + 1):
                        if (ix-center_x)**2+(iy-center_y)**2+(iz-center_z)**2<=r**2 :
                            ONE[iz, iy, ix, c] = 1 
    return ONE

# 相似度IoU评分函数
def iou_score(output, target):
    smooth = 1e-5
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return (intersection + smooth) / (union + smooth)

# 边界点图生成函数(3D)
def pointmap(target, num, times, radius):
    n = num
    t = times
    r = radius
    best_iou = 0

    edges = canny_3d(target)
    bound_data = edges.astype(np.int16)

    for i in range(t):
        points, index = point_select(bound_data, target, n)
        npindex = np.array(index)
        M = Griddata(npindex)
        iou = iou_score(M, target)
        if iou > best_iou:
            best_iou = iou
            best_points = points
            best_index = npindex

    print(best_iou)
    out = point_circle(best_index, r)
    return out#best_index #, best_index,

# 拆分通道
def channel_split(mask):
    MaskArray = np.zeros((155, 240, 240, 3), np.uint8)
    wt_mask = np.zeros([155, 240, 240])
    wt_mask[mask == 1] = 1.
    wt_mask[mask == 2] = 1.
    wt_mask[mask == 4] = 1.
    tc_mask = np.zeros([155, 240, 240])
    tc_mask[mask == 1] = 1.
    tc_mask[mask == 2] = 0.
    tc_mask[mask == 4] = 1.
    et_mask = np.zeros([155, 240, 240])
    et_mask[mask == 1] = 0.
    et_mask[mask == 2] = 0.
    et_mask[mask == 4] = 1.

    MaskArray[:, :, :, 0] = wt_mask[:,:,:]
    MaskArray[:, :, :, 1] = tc_mask[:,:,:]
    MaskArray[:, :, : ,2] = et_mask[:,:,:]
    return MaskArray

# 还原成nii格式
def r_nii(points):
    PointMap = np.zeros([155,240,240])
    color = [2,1,4]
    for c in range(3):
        circle_points = np.argwhere(points[:,:,:,c] != 0)
        circle_points.tolist()
        circle_index = np.array(circle_points)
        for i in circle_index:
            ix = i[2]
            iy = i[1]
            iz = i[0]
            PointMap[iz,iy,ix] = points[iz,iy,ix,c]*color[c]
    return PointMap

def file_name_path(file_dir, dir=False, file=True):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

# if not os.path.exists(output_MGT):
#     print("输出文件夹 不存在")
#     os.makedirs(output_MGT)
#     print("MGT 输出目录创建成功")

# path_list = file_name_path(file_path)
# print(len(path_list))

root = r"E:\DATASETS\aaa\brats2017\Training" # 数据集路径
paths = glob(os.path.join(root, "*GG", "Brats17*"))
print(len(paths)) 

for path in tqdm(paths):
    name = os.path.basename(path) # 文件夹名(病例序号)
    print(name)
    file_path= os.path.join((path), name + "_seg.nii.gz") # 文件地址
    save_path = os.path.join((path), name + "_pm.nii.gz") # 保存路径
    file = sitk.ReadImage(file_path, sitk.sitkUInt8)
    mask = sitk.GetArrayFromImage(file) 
    MaskArray = channel_split(mask) # 通道拆分
    num = np.array([300, 200, 100]) # 定义wt、tc、et选点数
    points = pointmap(MaskArray, num, 50, 1) # 掩膜数组，边界点数量，迭代次数t，半径r
    PointMap = r_nii(points) # 恢复成nii格式
    saveout = sitk.GetImageFromArray(PointMap)
    sitk.WriteImage(saveout, save_path)

print("Done")