# Author: Zhenyu Shi
# Date: 09/10/2023
# update: 09/10/2023

from argparse import ArgumentParser
import numpy as np
import os

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures.bbox_3d import Box3DMode, CameraInstance3DBoxes, LiDARInstance3DBoxes

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pcd', help='Point cloud file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='../work_dirs/results', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # args.pcd = './data/kitti/000008.bin'
    args.config = '../configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py'
    args.checkpoint = '../checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-3class_mine.pth'
    return args


# def main(args):
#     # TODO: Support inference of point cloud numpy file.
#     # build the model from a config file and a checkpoint file
#     model = init_model(args.config, args.checkpoint, device=args.device)
#
#     # test a single point cloud sample
#     result, data = inference_detector(model, args.pcd)
#     bbox_lidar = result.pred_instances_3d.bboxes_3d
#     score = result.pred_instances_3d.scores_3d
#
#     # for BEV
#     # convert bbox from LiDARInstance3DBoxes to CameraInstance3DBoxes
#     bbox_cam = Box3DMode.convert(bbox_lidar, Box3DMode.LIDAR, Box3DMode.CAM)
#
#     # filter bbox by using score threshold
#     bbox_cam = bbox_cam.cpu().numpy()
#     score = score.cpu().numpy()
#     mask = score > args.score_thr
#     bbox_cam = bbox_cam[mask]
#
#     # save the results
#     save_tag = '{}_pred_bbox.npy'.format(args.pcd.split('/')[-1].split('.')[0])
#     np.save(os.path.join(args.out_dir, save_tag), bbox_cam)
#     print('The prediction result is saved in {}'.format(os.path.join(args.out_dir, save_tag)))


def main(args):
    # TODO: Support inference of point cloud numpy file.
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # test a multi point cloud sample
    sample_idx_list = [25, 63, 114, 127, 134, 40, 43, 46, 110]
    for idx in sample_idx_list:
        pcd = '../data/kitti/training/velodyne_reduced/{:06d}.bin'.format(idx)
        result, data = inference_detector(model, pcd)
        bbox_lidar = result.pred_instances_3d.bboxes_3d
        score = result.pred_instances_3d.scores_3d


        # filter bbox by using score threshold
        bbox_lidar = bbox_lidar.cpu().numpy()
        score = score.cpu().numpy()
        mask = score > args.score_thr
        bbox_lidar = bbox_lidar[mask]

        # save the results
        save_tag = '{}_pred_bbox_lidar.npy'.format(pcd.split('/')[-1].split('.')[0])
        np.save(os.path.join(args.out_dir, save_tag), bbox_lidar)
        print('The prediction result is saved in {}'.format(os.path.join(args.out_dir, save_tag)))



if __name__ == '__main__':
    args = parse_args()
    main(args)
