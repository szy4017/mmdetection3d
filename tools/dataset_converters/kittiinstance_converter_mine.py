from os import path as osp
from pathlib import Path

import mmengine

total_num = {
    0: 750,
    1: 750,
    2: 750,
    3: 750,
    4: 750,
    5: 750,
    6: 750,
    7: 750,
    8: 750,
    9: 731,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
}
fold_split = {
    'train': [0, 1, 2, 3, 4, 5, 6, 7, 9],
    'val': [8],
    'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}
split_list = ['train', 'valid', 'test']


def get_kittiinstance_info(split):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'KITTIInstance'},
            'data_list': {
                00000: {
                    'lidar_points':{
                        'lidat_path':'sequences/00/velodyne/000000.bin'
                    },
                    'pts_semantic_mask_path':
                        'sequences/000/labels/000000.labbel',
                    'sample_id': '00'
                },
                ...
            }
        }
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='KITTIInstance')
    data_list = []
    for i_folder in fold_split[split]:
        for j in range(0, total_num[i_folder]):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                    osp.join('sequences',
                             str(i_folder).zfill(2), 'velodyne',
                             str(j).zfill(6) + '.bin'),
                    'num_pts_feats':
                    4
                },
                'pts_semantic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         str(j).zfill(6) + '.label'),
                'sample_id':
                str(i_folder) + str(j)
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_kittiinstance_info_file(pkl_prefix, save_path):
    """Create info file of KITTIInstance dataset.

    Directly generate info file without raw data.

    Args:
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print('Generate info.')
    save_path = Path(save_path)

    kittiinstance_infos_train = get_kittiinstance_info(split='train')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'KITTIInstance info train file is saved to {filename}')
    mmengine.dump(kittiinstance_infos_train, filename)
    kittiinstance_infos_val = get_kittiinstance_info(split='val')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'KITTIInstance info val file is saved to {filename}')
    mmengine.dump(kittiinstance_infos_val, filename)
    kittiinstance_infos_test = get_kittiinstance_info(split='test')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'KITTIInstance info test file is saved to {filename}')
    mmengine.dump(kittiinstance_infos_test, filename)
