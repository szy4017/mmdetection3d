# Author: Zhenyu Shi
# Date: 09/01/2023
# update: 09/01/2023

from argparse import ArgumentParser

from mmdet3d.apis import inference_segmentor, init_model
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pcd', help='Point cloud file',
                        default='../points/point_feature_4.bin')
    parser.add_argument('--semantic_config', help='Config file for semantic augmentation',
                        default='./config_checkpoint/minkunet34v2_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti.py')
    parser.add_argument('--semantic_checkpoint', help='Checkpoint file for semantic augmentation',
                        default='./config_checkpoint/minkunet34v2_w32_torchsparse_8xb2-amp-laser-polar-mix-3x_semantickitti_20230510_221853-b14a68b3.pth')
    parser.add_argument('--instance_config', help='Config file for instance augmentation',
                        default='./config_checkpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_kitti-3d_mine.py')
    parser.add_argument('--instance_checkpoint', help='Checkpoint file for instance augmentation',
                        default='./config_checkpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_kitti_20230831_164304_mine.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    args = parser.parse_args()


    return args


def main(args):
    # build the model from a config file and a checkpoint file
    sem_model = init_model(args.semantic_config, args.semantic_checkpoint, device=args.device)
    ins_model = init_model(args.instance_config, args.instance_checkpoint, device=args.device)

    default_mask = np.zeros_like(args.pcd)
    default_label_mapping = np.load('../points/label_mapping.npy')
    sem_res, sem_data = inference_segmentor(sem_model, args.pcd, default_mask, default_label_mapping)
    sem_mask = sem_res.pred_pts_seg.pts_semantic_mask.cpu().numpy()
    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)