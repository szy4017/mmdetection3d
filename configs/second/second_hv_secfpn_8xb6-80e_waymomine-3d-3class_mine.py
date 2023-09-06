_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/waymomine-3d-3class_mine.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# checkpoint cfg
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

# training schedule for 1x
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)