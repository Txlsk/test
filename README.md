## 一、概述
1. 训练

（1）对 normal 和 restricted 两部分数据进行分类器训练，利用keras的ResNet50进行迁移训练出二分类模型。
（2）利用 cascade_rcnn 对restricted的图片进行检测训练，训练出可以检测识别5种违禁物品的模型。

2.推断

（1）利用二分类模型对测试集进行分类，将测试集中的图片分为normal和restricted两类。
（2）利用检测模型，对分类好的测试集进行5种违禁物品的检测。将最后的结果保存为json文件。

## 二、环境配置
CUDA:cuda_9.0
CUDNN:cudnn7.3.0
Anadconda:Anaconda3-5.2

创建conda虚拟环境
### （1）tensorflow环境配置

### (2) mmdetection环境配置
1、新建conda环境 
conda create -n mm python=3.6 && source activate py36

2、安装pytorch  
conda install pytorch torchvision -c pytorch

3、安装依赖项 mmcv 
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install .

conda install cython

4、获取mmdetection 
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

5、编译安装                                    
./compile.sh && python setup.py install

## 三、数据
解压数据
切换文件夹 cd ./data
运行 ./unzip_data.sh
test_b密码：I3sR5Ze4Yh57

./data/test_b_cls 文件夹下用于存放经过二分类后的测试集图片

## 四、训练
(1) 分类训练
激活 tf 环境：source activate tf
切换文件夹 cd ./code/cls_train
运行 ./cls_train.sh
模型保存在此文件夹下

（2）检测训练
激活 mmdetection 环境：source activate mm
切换文件夹 cd ./code/obj_train
运行 ./obj_train.sh
模型保存在 work_dirs 文件夹下

## 五、预测
(1) 分类预测
激活 tf 环境：source activate tf
切换文件夹 cd ./infer
运行 ./cls_infer.sh
分类图片保存于 ./data/test_b_cls

(2) 检测预测
激活 mmdetection 环境：source activate mm
切换文件夹 cd ./infer
运行 ./train_infer.sh
json文件保存于 ./submit













