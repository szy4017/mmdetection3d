import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def load_data(point_path, gt_mask_path, pred_mask_path):
    """
    读取文件
    Args:
        point_path:
        gt_mask_path:
        pred_mask_path:

    Returns:

    """
    point = np.load(point_path)
    gt_mask = np.load(gt_mask_path)
    pred_mask = np.load(pred_mask_path)
    return point[:, :3], gt_mask, pred_mask


def draw_scene(point, mask=None):
    """
    根据mask绘制点云场景
    Args:
        point:
        mask:

    Returns:

    """
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 设置点云坐标
    point_cloud.points = o3d.utility.Vector3dVector(point)

    # 设置点云颜色
    colors = np.zeros((len(point), 3))  # 创建与点云大小相同的颜色数组

    # 定义颜色映射（这里使用19种不同的颜色）
    color_map = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1],
                 [1, 0, 1], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0],
                 [0, 0.5, 0.5], [0.5, 0, 0.5], [0.75, 0.25, 0], [0.75, 0, 0.25],
                 [0, 0.75, 0.25], [0.25, 0.75, 0], [0.25, 0, 0.75], [0, 0.25, 0.75],
                 [0.5, 0.25, 0.25]]
    if mask is not None:
        for i, category in enumerate(np.unique(mask)):
            colors[mask == category] = color_map[i]  # 根据类别设置颜色
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])


if __name__ == '__main__':
    # point_path = '../points/point.npy'
    # gt_mask_path = '../points/gt_mask.npy'
    # pred_mask_path = '../points/pred_mask.npy'
    #
    # point, gt_mask, pred_mask = load_data(point_path, gt_mask_path, pred_mask_path)
    # # draw_scene(point, gt_mask)  # 绘制gt场景
    # draw_scene(point, pred_mask)    # 绘制pred场景

    point_path = '../points/waymo_point.bin'
    point = np.fromfile(point_path, dtype=np.float32).reshape([-1, 6])
    draw_scene(point[:, :3])