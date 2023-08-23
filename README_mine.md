### 对于waymo数据集转换的说明
我们使用mmdet3d中的create_data.py方法实现waymo数据转换成kitti format

具体操作详见DOS：[https://mmdetection3d.readthedocs.io/zh_CN/latest/advanced_guides/datasets/waymo.html][waymo2kitti]

[waymo2kitti]: https://mmdetection3d.readthedocs.io/zh_CN/latest/advanced_guides/datasets/waymo.html

使用转换命令
```shell script
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
```
目前只采用了waymo部分训练数据进行数据转换，转换后得到如下的文件结构
```text
waymo
|-- kitti_format
|   |-- trainging
|   |   |-- calib 
|   |   |-- cam_sync_label_0
|   |   |-- cam_sync_label_1
|   |   |-- cam_sync_label_2
|   |   |-- cam_sync_label_3
|   |   |-- cam_sync_label_4
|   |   |-- cam_sync_label_all
|   |   |-- image_0     # camera FRONT
|   |   |-- image_1     # camera FRONT_LEFT
|   |   |-- image_2     # camera FRONT_RIGHT
|   |   |-- image_3     # camera SIDE_LEFT
|   |   |-- image_4     # camera SIDE_RIGHT
|   |   |-- image_all
|   |   |-- label_0
|   |   |-- label_1
|   |   |-- label_2
|   |   |-- label_3
|   |   |-- label_4
|   |   |-- label_all
|   |   |-- pose
|   |   |-- timestamp
|   |   |-- velodyne
```
![waymo数据集传感器分布图](resources/waymo_sensor.png)
根据waymo数据集的传感器分布，选择`image_0`作为图像输入，对应的标签为`label_0`。

---

### 构建waymo_mine数据集
选择waymo数据集中的一部分子集作为waymo_mine，waymo-mine采用kitti format，数据筛选条件如下：
```text
location == ''
weather == 'sunny'
time_of_day == 'day'
```
waymo_mine包含19134张图像数据，及其对应的label，calib和velodyne。
数据划分，训练数据15307个，验证数据3827个。
并且，waymo_mine中点云的特征数量是6。

使用数据生成命令
```shell script
python tools/create_data_mine.py waymomine --root-path ./data/waymo_mine --out-dir ./data/waymo_mine --extra-tag waymo_mine
```
完成数据包的生成后，得到如下的文件结构
```text
waymo_mine
|-- ImageSets
|-- training
|-- waymo_mine_infos_test.pkl
|-- waymo_mine_infos_train.pkl
|-- waymo_mine_infos_trainval.pkl
|-- waymo_mine_infos_val.pkl
```
由于没有设置test部分，`waymo_mine_infos_test.pkl`应该是没有信息的，主要使用的是`waymo_mine_infos_train.pkl`,`waymo_mine_infos_trainval.pkl`,`waymo_mine_infos_val.pkl`。
这几个数据包是用于模型在训练和测试过程中进行数据加载用的。

---

### 使用SECOND模型在waymo_mine数据集上进行测试
#### 配置模型config文件的基本过程
配置文件一共包含4个部分：
1. 总配置文件
2. model配置文件
3. dataset配置文件
4. schedule配置文件

举例second模型的配置
1. 总配置文件
```python
_base_ = [
    '../_base_/models/second_hv_secfpn_kitti.py',
    '../_base_/datasets/waymomine-3d-3class_mine.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
```
在总配置文件中引用了model配置，dataset配置，schdule配置。
当创建自己的总配置文件时，需要相应的创建并引用这些配置文件。
配置文件的创建可以参考原有的形式，并在相应字段上进行一些修改。

2. model配置文件
由于没有对模型配置进行改动，所以延用了原有的`second_hv_secfpn_kitti.py`配置文件。

3. dataset配置文件
基于`kitti-3d-3class.py`配置文件进行修改，得到`waymomine-3d-3class_mine.py`。
主要修改了以下下字段
```python
data_root = 'data/waymo_mine/'
load_dim = 6
ann_file = 'waymo_mine_infos_train.pkl'
ann_file = 'waymo_mine_infos_val.pkl'
ann_file = 'waymo_mine_infos_val.pkl'
```

4. schedule配置文件
主要用于配置模型的训练过程，没有进行改动，延用原有的`cyclic-40e.py`和`default_runtime.py`。

#### 模型测试过程
使用以下命令进行模型测试
```shell script
python tools/test.py configs/second/second_hv_secfpn_8xb6-80e_waymomine-3d-3class_mine.py checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth
```
得到测试结果如下：
```text
----------- AP11 Results ------------

Pedestrian AP11@0.50, 0.50, 0.50:
bbox AP11:0.1287, 0.1289, 0.1289
bev  AP11:0.1515, 0.1510, 0.1510
3d   AP11:0.0832, 0.0821, 0.0821
Pedestrian AP11@0.50, 0.25, 0.25:
bbox AP11:0.1287, 0.1289, 0.1289
bev  AP11:0.4737, 0.4740, 0.4740
3d   AP11:0.4344, 0.4352, 0.4352
Cyclist AP11@0.50, 0.50, 0.50:
bbox AP11:0.0708, 0.0686, 0.0686
bev  AP11:0.0213, 0.0206, 0.0206
3d   AP11:0.0213, 0.0206, 0.0206
Cyclist AP11@0.50, 0.25, 0.25:
bbox AP11:0.0708, 0.0686, 0.0686
bev  AP11:0.0638, 0.0618, 0.0618
3d   AP11:0.0267, 0.0261, 0.0261
Car AP11@0.70, 0.70, 0.70:
bbox AP11:0.2626, 0.2623, 0.2623
bev  AP11:0.0763, 0.0761, 0.0761
3d   AP11:0.0340, 0.0339, 0.0339
Car AP11@0.70, 0.50, 0.50:
bbox AP11:0.2626, 0.2623, 0.2623
bev  AP11:0.1665, 0.1664, 0.1664
3d   AP11:0.0839, 0.0837, 0.0837

Overall AP11@easy, moderate, hard:
bbox AP11:0.1541, 0.1533, 0.1533
bev  AP11:0.0831, 0.0826, 0.0826
3d   AP11:0.0462, 0.0455, 0.0455

----------- AP40 Results ------------

Pedestrian AP40@0.50, 0.50, 0.50:
bbox AP40:0.1040, 0.1027, 0.1027
bev  AP40:0.1131, 0.1108, 0.1108
3d   AP40:0.0415, 0.0411, 0.0411
Pedestrian AP40@0.50, 0.25, 0.25:
bbox AP40:0.1040, 0.1027, 0.1027
bev  AP40:0.2702, 0.2702, 0.2702
3d   AP40:0.2475, 0.2468, 0.2468
Cyclist AP40@0.50, 0.50, 0.50:
bbox AP40:0.0267, 0.0259, 0.0259
bev  AP40:0.0037, 0.0036, 0.0036
3d   AP40:0.0044, 0.0042, 0.0042
Cyclist AP40@0.50, 0.25, 0.25:
bbox AP40:0.0267, 0.0259, 0.0259
bev  AP40:0.0271, 0.0264, 0.0264
3d   AP40:0.0130, 0.0127, 0.0127
Car AP40@0.70, 0.70, 0.70:
bbox AP40:0.0722, 0.0721, 0.0721
bev  AP40:0.0210, 0.0209, 0.0209
3d   AP40:0.0093, 0.0093, 0.0093
Car AP40@0.70, 0.50, 0.50:
bbox AP40:0.0722, 0.0721, 0.0721
bev  AP40:0.0458, 0.0457, 0.0457
3d   AP40:0.0231, 0.0230, 0.0230

Overall AP40@easy, moderate, hard:
bbox AP40:0.0676, 0.0669, 0.0669
bev  AP40:0.0459, 0.0451, 0.0451
3d   AP40:0.0184, 0.0182, 0.0182
```
主要评价指标有AP11和AP40，他们的主要区别在于PR曲线采样点的数量不同。
AP11的采样点为11个，AP40的采样点为40个。
AP40更能准确反应算法的检测性能，一般情况下AP40的数值要小于AP11。