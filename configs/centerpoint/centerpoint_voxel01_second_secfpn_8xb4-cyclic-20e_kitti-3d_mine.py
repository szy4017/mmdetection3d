_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/models/centerpoint_voxel01_second_secfpn_kitti_mine.py',
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))