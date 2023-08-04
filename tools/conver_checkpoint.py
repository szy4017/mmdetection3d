import torch

if __name__ == '__main__':
    load_ckpt_path_mine = '../checkpoints/epoch_1.pth'
    load_ckpt_path_save = '../checkpoints/minkunet34_w32_spconv_8xb2-laser-polar-mix-3x_semantickitti_20230512_233817-72b200d8.pth'
    save_path = '../checkpoints/munkunet_kittiinstance_20230715_1829.pth'

    ckpt_mine = torch.load(load_ckpt_path_mine)
    ckpt_save = torch.load(load_ckpt_path_save)
    ckpt_save = ckpt_mine['state_dict']
    for key in ckpt_save.keys():
        print(key)
        param = ckpt_save[key]
        if len(param.shape) == 5:
            ckpt_save[key] = param.permute(1, 2, 3, 4, 0)
    torch.save(ckpt_save, save_path)
    pass