# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from tools.dataset_converters import kittiinstance_converter_mine


def kittiinstance_data_prep(info_prefix, out_dir):
    """Prepare the info file for SemanticKITTI dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
    """
    kittiinstance_converter_mine.create_kittiinstance_info_file(
        info_prefix, out_dir)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='Whether to only generate ground truth database.')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset == 'kittiinstance':
        kittiinstance_data_prep(
            info_prefix=args.extra_tag, out_dir=args.out_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
