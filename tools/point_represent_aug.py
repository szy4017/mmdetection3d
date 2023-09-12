# Author: Zhenyu Shi
# Date: 09/01/2023
# update: 09/10/2023

from argparse import ArgumentParser
import pickle
from mmdet3d.apis import inference_detector, inference_segmentor, init_model
import numpy as np
from mmdet3d.structures.bbox_3d import Box3DMode, CameraInstance3DBoxes, LiDARInstance3DBoxes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pcd', help='Point cloud file',
                        default='../demo/data/kitti/000008.bin')
    parser.add_argument('--semantic_config', help='Config file for semantic augmentation',
                        default='../configs/minkunet/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_kittiinstanceseg_mine.py')
    parser.add_argument('--semantic_checkpoint', help='Checkpoint file for semantic augmentation',
                        default='../work_dirs/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_kittiinstanceseg_mine/epoch_10.pth')
    parser.add_argument('--instance_config', help='Config file for instance augmentation',
                        default='../configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_kitti-3d_mine.py')
    parser.add_argument('--instance_checkpoint', help='Checkpoint file for instance augmentation',
                        default='../work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_kitti-3d_mine/epoch_20.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    args = parser.parse_args()


    return args


def main(args):
    # build the model from a config file and a checkpoint file
    # sem_model = init_model(args.semantic_config, args.semantic_checkpoint, device=args.device)
    ins_model = init_model(args.instance_config, args.instance_checkpoint, device=args.device)

    # semantic representation
    # default_mask = np.zeros((20000)).astype(np.uint32)
    # default_label_mapping = np.load('../points/label_mapping.npy')
    # sem_res, sem_data = inference_segmentor(sem_model, args.pcd, default_mask, default_label_mapping)
    # sem_mask = sem_res.pred_pts_seg.pts_semantic_mask.cpu().numpy()
    # print('sem_mask_info: dtype: {}, shape: {}'.format(sem_mask.dtype, sem_mask.shape))
    # print('save semantic representation result')
    # np.save('../points/point_kitti_000575.npy', sem_data['inputs']['points'].numpy())
    # np.save('../points/pred_mask_kitti_000575.npy', sem_mask)
    # print('finished!')

    # instance representation
    ins_res, ins_data = inference_detector(ins_model, args.pcd)
    # print(ins_res.pred_instances_3d.bboxes_3d)
    LiDARBBOX = ins_res.pred_instances_3d.bboxes_3d
    CameraBBOX = Box3DMode.convert(LiDARBBOX, Box3DMode.LIDAR, Box3DMode.CAM)
    # print(CameraBBOX)
    ins_cam_bbox = CameraBBOX.cpu().numpy()
    ins_center = ins_res.pred_instances_3d.bboxes_3d.center.cpu().numpy()
    ins_range = ins_res.pred_instances_3d.bboxes_3d.corners.cpu().numpy()
    ins_score = ins_res.pred_instances_3d.scores_3d.cpu().numpy()
    print('ins_center_info: dtype: {}, shape: {}'.format(ins_center.dtype, ins_center.shape))
    print('ins_range_info: dtype: {}, shape: {}'.format(ins_range.dtype, ins_range.shape))
    print('ins_score_info: dtype: {}, shape: {}'.format(ins_score.dtype, ins_score.shape))
    print('ins_cam_bbox_info: dtype: {}, shape: {}'.format(ins_cam_bbox.dtype, ins_cam_bbox.shape))
    # 根据score thr筛选
    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)