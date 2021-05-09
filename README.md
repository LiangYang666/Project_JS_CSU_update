# 基于Tianxiaomo/pytorch-YOLOv4
https://github.com/Tianxiaomo/pytorch-YOLOv4
添加损失函数等注释


## 所需环境
### requirement.txt


## 文件夹分布
```

├── data                      数据存放文件夹 
│   ├── img_1080p.py          1080p数据集
│       ├── _classes.txt      类别
│       ├── val.txt           测试集
│       ├── train_xyb_orgin.txt    训练集
│       ├── checkpoints       初始权重存储
│       ├── checkpoints2      小样本权重存储
│       ├── img_1080p         图片文件夹
│       ├── img_test          小样本权重存储
│       ├── out_img         测试集识别后的图像带框输出位置
│       ├── output          测试集的识别准确度及mAP
│       ├── txt_file_dect      测试集识别后后生成的标签
│       └── txt_file_truth     测试集真实标签
├── ProjectJS_CSU_Update          项目工程
│   ├── cfg.py              配置
│   ├── convert_pytorchYOLOV4_mAP.py 转换标签文件
│   ├── dataset.py                  数据集加载有关
│   ├── mAP.py                      测试mAP
│   ├── modeles_YL.py               识别模型
│   ├── train_YL.py                 训练
│   ├── utils.py                    检测工具
│   └── test_cmd.py                 cmd

```


## 测试
### 训练和识别 test_cmd.py


