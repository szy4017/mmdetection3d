import numpy as np
from mmengine import load
import matplotlib.pyplot as plt
import cv2

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes


def draw_box(box_info_file, flag='gt'):
    if box_info_file.endswith('pkl'):
        info_file = load(box_info_file)
        bboxes_3d = []
        for instance in info_file['data_list'][0]['instances']:
            bboxes_3d.append(instance['bbox_3d'])
        gt_bboxes_3d = np.array(bboxes_3d, dtype=np.float32)
    elif box_info_file.endswith('npy'):
        gt_bboxes_3d = np.load(box_info_file)
    print(gt_bboxes_3d)
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d)

    visualizer = Det3DLocalVisualizer()
    # set bev image in visualizer
    visualizer.set_bev_image()
    print(gt_bboxes_3d.bev)
    # draw bev bboxes
    if flag == 'gt':
        visualizer.draw_bev_bboxes(gt_bboxes_3d, scale=10, edge_colors='green')
    elif flag == 'pred':
        visualizer.draw_bev_bboxes(gt_bboxes_3d, scale=10, edge_colors='orange')
    visualizer.show(wait_time=1)
    visualizer.close()


def draw_point(point_file):
    # 点云读取
    pointcloud = np.fromfile(str(point_file), dtype=np.float32, count=-1).reshape([-1, 4])
    # 设置鸟瞰图范围
    side_range = (-35, 35)  # 左右距离
    fwd_range = (0, 100)  # 后前距离

    x_points = pointcloud[:, 0]
    y_points = pointcloud[:, 1]
    z_points = pointcloud[:, 2]

    # 获得区域内的点
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    res = 0.1  # 分辨率0.05m
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    print(x_img.min(), x_img.max(), y_img.min(), x_img.max())

    # 填充像素值
    height_range = (-2, 0.5)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])


    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)


    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])

    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value

    # imshow （灰度）
    plt.imshow(im)

    # 隐藏坐标轴
    plt.axis('off')

    # 设置图片边界与内容紧密适配
    plt.tight_layout()

    plt.show()

    save_tag = point_file.split('/')[-1].split('.')[0]+'_bev_point.png'
    cv2.imwrite('points/bev_results/{}'.format(save_tag), im)
    print('The BEV point image is saved in points/bev_results/{}'.format(save_tag))


if __name__ == '__main__':
    # box_info_file = 'demo/data/kitti/000008.pkl'
    data_idx = 7396
    pred_box_info_file = 'work_dirs/results/{:06d}_pred_bbox_m.npy'.format(data_idx)
    gt_box_info_file = 'points/{:06d}_gt_box.npy'.format(data_idx)
    point_file = 'data/kitti/training/velodyne/{:06d}.bin'.format(data_idx)
    draw_box(pred_box_info_file, flag='pred')
    draw_box(gt_box_info_file)
    draw_point(point_file)