import os
from tqdm import tqdm


def txt_images_to_file(fpath='../data/fangzhen/'):
    '''将某文件txt中的文件名对应的照片复制至'''
    from shutil import copyfile
    txt = 'val.txt'
    txt_file = fpath + txt
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    out_path = fpath + 'test_img/'
    if not os.path.exists(out_path):  # 判断文件夹是否存在
        os.makedirs(out_path)
    for i,line in enumerate(lines):
        imgfile = line.split()[0]
        imgfile = fpath + imgfile
        copyfile(imgfile, out_path+imgfile.split('/')[-1])


# txt_images_to_file()

# rootdir = os.path.join(os.getcwd(),"scripts/enum_parser.py")
# rootdir += " %s"


# 训练

# cmd = 'python train_YL.py -g 1 -pretrained yolov4.conv.137.pth -classes 5 -dir ../data/fangzhen -tlp train10.txt -epoch 1000'
# cmd = 'python train_YL.py - g 1 -pretrained yolov4.conv.137.pth -classes 5 -dir.. / data / fangzhen - tlp train_90_10.txt - epoch 1000'
# cmd = 'python train_YL.py -g 0 -pretrained yolov4.conv.137.pth -classes 2 -dir ../data/FLIR -tlp train.txt -epoch 1000'
# cmd = 'python train_YL.py -g 0 -pretrained NWPU_epoch200.pth -classes 3 -dir ../data/img_1080p -tlp train_xyb_orgin.txt -epoch 200'
# cmd = 'python train_YL.py -g 0 -pretrained NWPU_epoch200.pth -classes 3 -dir ../data/img_1080p -tlp train_xyb_aug.txt -ckp_path checkpoints2 -epoch 10000'


# 测试val.txt 索引的图片 输出至out_img 文件夹内 这将会自动计算mAP
# cmd = 'python models_YL.py 5 ../data/fangzhen ckp_fangzhen_5c_100_8_ep200.pth val.txt out_img'
# cmd = 'python models_YL.py 5 ../data/fangzhen ckp_fangzhen_5c_100_8_ep200.pth val.txt out_img'
# cmd = 'python models_YL.py 36 ../data/fangzhen Yolov4_fangzhen_epoch30.pth val.txt out_img'
# cmd = 'python models_YL.py 3 ../data/img_1080p Yolov4_epoch250.pth val.txt out_img'
# cmd = 'python models_YL.py 2 ../data/FLIR Yolov4_FLIR_2c_epoch800.pth train.txt out_img'

# 测试img_test1 内的图片
# cmd = 'python models_YL.py 5 ../data/fangzhen Yolov4_fangzhen_5c_epoch400.pth img_test'
# 测试img_test1内的图片 预测得到的带框图输出至img_test1_pre中 预测的txt输出至img_test1_txt中 img_test1文件名可改
# cmd = 'python models_YL.py 5 ../data/fangzhen Yolov4_fangzhen_5c_epoch700.pth img_test'
# cmd = 'python models_YL.py 3 ../data/img_1080p Yolov4_epoch250.pth img_test'

# 红外测试img_test1 内的图片
# cmd = 'python models_YL.py 2 ../data/FLIR Yolov4_FLIR_2c_epoch1000.pth img_test8'

# cmd = 'pythofangzhen Yolov4_fangzhen_5c_epoch10.pthn models_YL.py 80 ../data/coco yolov4.weights img_test'
# cmd = 'python models_YL.py 80 ../data/coco Yolov4_coco.pth img_test'
# cmd = 'python models_YL.py 80 ../data/coco yolov4.conv.137.pth img_test'
# cmd = 'python models_YL.py 80 ../data/coco Yolov4_epoch1.pth img_test'
# cmd = 'python models_YL.py 80 ../data/coco yolov4137.pth img_test'


# 测试mAP 源文件为txt_file_dect 和 txt_file_truth
# cmd = 'python mAP.py -dir ../data/img_1080p -ckpn 1'
# cmd = 'python mAP.py -dir ../data/fangzhen -ckpn 1000'
#

# os.system(cmd)
some_ckp1 = ['100_8_ep800', '100_8_ep900', '100_8_ep1000']


some_ckp2 = ['l0001_100_10_ep200', 'l0001_100_10_ep250',
             'l0001_100_10_ep300', 'l0001_100_10_ep350',
             'l0001_100_10_ep400', 'l0001_100_10_ep500',

             'l001_100_10_ep200', 'l001_100_10_ep250',
             'l001_100_10_ep300', 'l001_100_10_ep350',
             'l001_100_10_ep400', 'l001_100_10_ep450',
             'l001_100_10_ep500',

             'l01_100_10_ep200', 'l01_100_10_ep250',
             'l01_100_10_ep300', 'l01_100_10_ep350',
             'l01_100_10_ep400', 'l01_100_10_ep450',
             'l01_100_10_ep500',

             'l1_100_10_ep200', 'l1_100_10_ep250',
             'l1_100_10_ep300', 'l1_100_10_ep350',
             'l1_100_10_ep400', 'l1_100_10_ep450',
             'l1_100_10_ep500',

             'l0005_100_10_ep200', 'l0005_100_10_ep250',
             'l0005_100_10_ep300', 'l0005_100_10_ep350',
             'l0005_100_10_ep400', 'l0005_100_10_ep450',
             'l0005_100_10_ep500',

             'l005_100_10_ep200', 'l005_100_10_ep250',
             'l005_100_10_ep300', 'l005_100_10_ep350',
             'l005_100_10_ep400', 'l005_100_10_ep450',
             'l005_100_10_ep500',

             'l05_100_10_ep200', 'l05_100_10_ep250',
             'l05_100_10_ep300', 'l05_100_10_ep350',
             'l05_100_10_ep400', 'l05_100_10_ep450',
             'l05_100_10_ep500',

             'l5_100_10_ep200', 'l5_100_10_ep250',
             'l5_100_10_ep300', 'l5_100_10_ep350',
             'l5_100_10_ep400', 'l5_100_10_ep450',
             'l5_100_10_ep500',
             ]
some_ckp3 = ['l5_100_10_ep125']
# for n in some_ckp3:
#     print(n)
#     # cmd = f'python models_YL.py 5 ../data/fangzhen ckp_fangzhen_5c_{n}.pth val.txt out_img'
#     cmd = f'python models_YL.py 5 ../data/fangzhen ckp_fangzhen_5c_{n}.pth img_test2'
#     os.system(cmd)
