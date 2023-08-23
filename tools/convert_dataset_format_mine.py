import numpy as np
import os
import random


def convert_point_file(source_path, target_path):
    """
    转换点云bin文件
    Args:
        source_path:
        target_path:

    Returns:

    """
    file_num = 7472
    split_num = 11

    # 构建文件的保存目录
    file_save_path_list = []
    dir_flag = -1
    for i in range(file_num):
        if i % 750 == 0:
            dir_flag = dir_flag + 1
            file_flag = 0
            dir_name = '{:02d}'.format(dir_flag)
            file_name = '{:06}'.format(file_flag)
            save_path = os.path.join(target_path, 'sequences', dir_name, 'velodyne', file_name+'.bin')
            file_save_path_list.append(save_path)
        else:
            file_flag = file_flag + 1
            dir_name = '{:02d}'.format(dir_flag)
            file_name = '{:06}'.format(file_flag)
            save_path = os.path.join(target_path, 'sequences', dir_name, 'velodyne', file_name+'.bin')
            file_save_path_list.append(save_path)


    data_file_list = os.listdir(os.path.join(source_path, 'point_feature'))
    data_file_list.sort()
    val_list = []
    path_list = []
    for i, df in enumerate(data_file_list):
        print(df)
        data_file = os.path.join(source_path, 'point_feature', df)
        data = np.load(data_file)
        data_f32 = data.astype(np.float32)
        data_f32_4 = data_f32[:, :4]
        # print(data_f32_4.shape)
        if data_f32_4.shape == (0, 4):
            print('no point')
            continue
        data_f32_4.tofile(file_save_path_list[i])
        # if '/08/' in file_save_path_list[i]:
        #     val_list.append(data_f32_4)
        #     path_list.append(df)
        # elif '/09/' in file_save_path_list[i]:
        #     return val_list, path_list


def convert_label_file(source_path, target_path):
    """
    转换标签label文件
    Args:
        source_path:
        target_path:

    Returns:

    """
    file_num = 7472
    split_num = 11

    # 构建文件的保存目录
    file_save_path_list = []
    dir_flag = -1
    for i in range(file_num):
        if i % 750 == 0:
            dir_flag = dir_flag + 1
            file_flag = 0
            dir_name = '{:02d}'.format(dir_flag)
            file_name = '{:06}'.format(file_flag)
            save_path = os.path.join(target_path, 'sequences', dir_name, 'labels', file_name+'.label')
            file_save_path_list.append(save_path)
        else:
            file_flag = file_flag + 1
            dir_name = '{:02d}'.format(dir_flag)
            file_name = '{:06}'.format(file_flag)
            save_path = os.path.join(target_path, 'sequences', dir_name, 'labels', file_name+'.label')
            file_save_path_list.append(save_path)


    label_file_list = os.listdir(os.path.join(source_path, 'point_sem_label'))
    instance_file_list = os.listdir(os.path.join(source_path, 'point_ins_label'))
    label_file_list.sort()
    instance_file_list.sort()
    val_list = []
    path_list = []
    for i, (labf, insf) in enumerate(zip(label_file_list, instance_file_list)):
        print(labf, ' ', insf)
        label_file = os.path.join(source_path, 'point_sem_label', labf)
        instance_file = os.path.join(source_path, 'point_ins_label', insf)
        label_data = np.load(label_file)
        instance_data = np.load(instance_file)
        label_data = label_data.astype(np.uint16)
        instance_data = instance_data.astype(np.uint16)
        data_uint32 = (instance_data.astype(np.uint32) << 16) | label_data
        data_uint32.tofile(file_save_path_list[i])
        # if '/08/' in file_save_path_list[i]:
        #     val_list.append(data_uint32)
        #     path_list.append(labf)
        # elif '/09/' in file_save_path_list[i]:
        #     return val_list, path_list


def waymo_mine(root):
    calib_path = os.path.join(root, 'calib')
    image_path = os.path.join(root, 'image_2')
    label_path = os.path.join(root, 'label_2')
    velodyne_path = os.path.join(root, 'velodyne')

    calib_list = os.listdir(calib_path)
    image_list = os.listdir(image_path)
    label_list = os.listdir(label_path)
    velodyne_list = os.listdir(velodyne_path)
    calib_list.sort()
    image_list.sort()
    label_list.sort()
    velodyne_list.sort()

    calib_name_list = []
    image_name_list = []
    label_name_list = []
    velodyne_name_list = []
    for ca in calib_list:
        calib_name_list.append(ca.split('.')[0])
    for im in image_list:
        image_name_list.append(im.split('.')[0])
    for la in label_list:
        label_name_list.append(la.split('.')[0])
    for ve in velodyne_list:
        velodyne_name_list.append(ve.split('.')[0])

    common_name_list = list(set(calib_name_list) & set(image_name_list) & set(label_name_list) & set(velodyne_name_list))
    remain_calib_name_list = [x for x in calib_name_list if x not in common_name_list]
    remain_image_name_list = [x for x in image_name_list if x not in common_name_list]
    remain_label_name_list = [x for x in label_name_list if x not in common_name_list]
    remain_velodyne_name_list = [x for x in velodyne_name_list if x not in common_name_list]

    if remain_calib_name_list is not []:
        for re_ca in remain_calib_name_list:
            try:
                os.remove(os.path.join(root, 'calib', re_ca+'.txt'))
                print(f"remove {re_ca}+.txt in calib")
            except FileNotFoundError:
                print(f"{re_ca}+.txt in calib not found")
    if remain_image_name_list is not []:
        for re_im in remain_image_name_list:
            try:
                os.remove(os.path.join(root, 'image_2', re_im+'.jpg'))
                print(f"remove {re_im}+.jpg in image_2")
            except FileNotFoundError:
                print(f"{re_im}+.jpg in image_2 not found")
    if remain_label_name_list is not []:
        for re_la in remain_label_name_list:
            try:
                os.remove(os.path.join(root, 'label_2', re_la+'.txt'))
                print(f"remove {re_la}+.txt in label_2")
            except FileNotFoundError:
                print(f"{re_la}+.txt in label_2 not found")
    if remain_velodyne_name_list is not []:
        for re_ve in remain_velodyne_name_list:
            try:
                os.remove(os.path.join(root, 'velodyne', re_ve+'.bin'))
                print(f"remove {re_ve}+.bin in velodyne")
            except FileNotFoundError:
                print(f"{re_ve}+.bin in velodyne not found")


def create_ImageSets_waymo_mine(root):
    file_list = os.listdir(os.path.join(root, 'calib'))
    name_list = []
    for f in file_list:
        name_list.append(f.split('.')[0])

    val_set = random.sample(name_list, 3827)
    train_set = [x for x in name_list if x not in val_set]
    val_set.sort()
    train_set.sort()

    with open(os.path.join(root, 'val.txt'), 'w') as file:
        for item in val_set:
            file.write("%s\n" % item)
        print('val set 创建成功')
    with open(os.path.join(root, 'train.txt'), 'w') as file:
        for item in train_set:
            file.write("%s\n" % item)
        print('train set 创建成功')

    pass


if __name__ == '__main__':
    # source_path = '/data/szy4017/data/kitti_instance/kitti_instance/training'
    # target_path = '/data/szy4017/data/kitti_instance_2'
    #
    # convert_point_file(source_path, target_path)
    # convert_label_file(source_path, target_path)

    # point_val_list, point_path_list = convert_point_file(source_path, target_path)
    # label_val_list, label_path_list = convert_label_file(source_path, target_path)
    #
    # for pv, lv, pp, lp in zip(point_val_list, label_val_list, point_path_list, label_path_list):
    #     print(pv.shape, ' ', lv.shape)
    #     if pv.shape[0] <= 86:
    #         print('point too small')
    #         print(pp, ' ', lp)


    # waymo_mine('/data/szy4017/data/waymo_mine/training')
    create_ImageSets_waymo_mine('/data/szy4017/data/waymo_mine/training')
