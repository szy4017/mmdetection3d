# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from tools.dataset_converters import kittiinstance_converter_mine
from tools.dataset_converters import waymomine_converter_mine as waymomine
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
from tools.dataset_converters.create_gt_database import create_groundtruth_database


def kittiinstance_data_prep(info_prefix, out_dir):
    """Prepare the info file for KITTIInstance dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
    """
    kittiinstance_converter_mine.create_kittiinstance_info_file(
        info_prefix, out_dir)


def waymomine_data_prep(root_path,
                        info_prefix,
                        version,
                        out_dir,
                        with_plane=False):
    """Prepare data related to waymo_mine dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    waymomine.create_kitti_info_file(root_path, info_prefix, with_plane)
    waymomine.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    # update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_test_path)
    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))



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
    elif args.dataset == 'waymomine':
        waymomine_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            with_plane=args.with_plane)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
