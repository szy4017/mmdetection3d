# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_segmentor, init_model
from mmdet3d.registry import VISUALIZERS
import numpy as np
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pcd', help='Point cloud file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
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


    idx = 63
    args.pcd = '../data/kitti/training/velodyne_reduced/{:06d}.bin'.format(idx)
    with open('../points/000000.label', 'rb') as file:
        data = file.read()
    mask = np.frombuffer(data, dtype=np.uint32)
    mask = mask & 0xFFFF
    args.mask = np.zeros_like(mask)
    args.label_mapping = np.load('../points/label_mapping.npy')
    args.pcd = '../points/point_target/000043_target_0.bin'
    args.config = '../configs/minkunet/minkunet34_w32_spconv_8xb2-amp-laser-polar-mix-3x_kittiinstanceseg_mine.py'
    # args.config = '../configs/minkunet/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti.py'
    args.checkpoint = '../work_dirs/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_kittiinstanceseg_mine/epoch_10.pth'
    # args.checkpoint = '../checkpoints/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti_20230512_233817-72b200d8.pth'

    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # test a single point cloud sample
    result, data = inference_segmentor(model, args.pcd, args.mask, args.label_mapping)
    pred_mask = result.pred_pts_seg.pts_semantic_mask.cpu().numpy()

    # save the results
    save_tag = '{}_pred_mask.npy'.format(args.pcd.split('/')[-1].split('.')[0])
    np.save(os.path.join(args.out_dir, save_tag), pred_mask)
    print('The prediction result is saved in {}'.format(os.path.join(args.out_dir, save_tag)))


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # test a multi point cloud sample
    # sample_idx_list = [25, 63, 114, 127, 134, 40, 43, 46, 110]
    sample_idx_list = [0, 1, 2, 3, 4, 5]
    for idx in sample_idx_list:
        # pcd = '../data/kitti/training/velodyne_reduced/{:06d}.bin'.format(idx)
        pcd = '../points/point_target/000046_target_{}.bin'.format(idx)
        result, data = inference_segmentor(model, pcd, args.mask, args.label_mapping)
        pred_mask = result.pred_pts_seg.pts_semantic_mask.cpu().numpy()

        # save the results
        save_tag = '{}_pred_mask.npy'.format(pcd.split('/')[-1].split('.')[0])
        np.save(os.path.join(args.out_dir, save_tag), pred_mask)
        print('The prediction result is saved in {}'.format(os.path.join(args.out_dir, save_tag)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
