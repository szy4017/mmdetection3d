# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_segmentor, init_model
from mmdet3d.registry import VISUALIZERS
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pcd', help='Point cloud file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()


    # args.pcd = '../points/000000.bin'
    args.pcd = '../points/point_feature_4.bin'
    # args.pcd = np.fromfile('../points/000000.bin', dtype=np.float32).reshape(-1, 4)
    with open('../points/000000.label', 'rb') as file:
        data = file.read()
    mask = np.frombuffer(data, dtype=np.uint32)
    mask = mask & 0xFFFF
    args.mask = np.zeros_like(mask)
    # args.mask = '../points/000000.label'
    args.label_mapping = np.load('../points/label_mapping.npy')
    args.config = '../configs/minkunet/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti.py'
    args.checkpoint = '../checkpoints/munkunet_kittiinstance_20230715_1829.pth'
    # args.checkpoint = '../checkpoints/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti_20230512_233817-72b200d8.pth'

    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single point cloud sample
    result, data = inference_segmentor(model, args.pcd, args.mask, args.label_mapping)

    point = data['inputs']['points'].numpy()
    gt_mask = args.label_mapping[args.mask]
    pred_mask = result.pred_pts_seg.pts_semantic_mask.cpu().numpy()
    # save the results
    print('save result')
    np.save('../points/point.npy', point)
    np.save('../points/gt_mask.npy', gt_mask)
    np.save('../points/pred_mask.npy', pred_mask)
    print('finished!')

    # show the results
    # points = data['inputs']['points']
    # data_input = dict(points=points)
    # visualizer.add_datasample(
    #     'result',
    #     data_input,
    #     data_sample=result,
    #     draw_gt=False,
    #     show=args.show,
    #     wait_time=-1,
    #     out_file=args.out_dir,
    #     vis_task='lidar_seg')


if __name__ == '__main__':
    args = parse_args()
    main(args)
