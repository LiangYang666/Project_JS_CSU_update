import os
import random
import cv2
from tqdm import tqdm
import time
import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
# from scipy.interpolate import spline
from scipy.interpolate import make_interp_spline

dataset_dir = '../data/fangzhen/'
file = dataset_dir + 'train_100_10.txt'



def sample_n_train(in_n=10, out_n=9):   # 对于仿真数据 抽取10张中的out_n张生成训练集
    # class_l = list(range(10))
    class_l = [[] for i in range(5)]
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        bbox = line.split()[1]
        class_number = int(bbox.split(',')[-1])
        class_l[class_number].append(line)

    for i in range(5):
        random.shuffle(class_l[i])
        class_l[i] = class_l[i][:out_n]
        # print(class_l[i])


    with open(dataset_dir+'train100_{}.txt'.format(out_n), 'w') as out_txt:
        for list_each in class_l:
            for line in list_each:
                out_txt.write(line)
# FileName
def ChangeTxt(FileName = file, NewFilepath = dataset_dir, r = 0.8):
    """读取数据函数,返回list类型的数据
    FileName为输入文件的名称
    NewFileName为希望生成的txt文件路径
    r = 0.8   缩放比例r默认0.8
    train_100_10.txt
    """
    txtname = os.path.basename(FileName)
    txtname = txtname.split('_')
    NewFileName = NewFilepath+txtname[0] + f'_{int(r*100)}_' + txtname[2]
    with open(FileName, 'r') as txtData:
        # 读取数据函数,返回list类型的数据
        lines = txtData.readlines()
    with open(NewFileName, 'w') as txtData_w:
        # pbar = tqdm(range(len(lines)))
        for line in lines:
            # pbar.update(1)
            lineData = line.split()  # 分割空白和\n
            # lineData3 = lineData[:]
            bbox = lineData[-1].split(',')

            bbox = [int(x) for x in bbox]
            xmin, ymin, xmax, ymax = bbox[:4]

            dx = (xmax-xmin)*(1-r)*0.5
            dy = (ymax-ymin)*(1-r)*0.5
            xmin += dx
            ymin += dy
            xmax -= dx
            ymax -= dy
            xmin = max(xmin, 0)
            xmax = min(xmax, 1920)
            ymin = max(ymin, 0)
            ymax = min(ymax, 1080)
            xmin, ymin, xmax, ymax =[round(x) for x in [xmin, ymin, xmax, ymax]]
            print('orgin :', line, end='')
            line = lineData[0] + ' {},{},{},{},{}\n'.format(xmin, ymin, xmax, ymax, bbox[-1])
            txtData_w.write(line)
            print('change:', line, end='')

def Txt_tool1():
    """待完善"""
    file = '../data/fangzhen/test_log (1).txt'
    dontwant = ('ARMT', 'CANNONT', 'FIGHTERT', 'HELIT', 'MISSILET')
    i_del_list = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(dontwant):
                i_del_list.append(i)


    with open(file, 'w') as f:
        for line in lines:
            f.write(line)

def cut_img_tool3():
    """裁剪图片 根据炫的框裁剪"""
    txt_file = '../data/fangzhen/val.txt'
    txt_class = '../data/fangzhen/_classes.txt'

    with open(txt_class, 'r') as f:
        class_name = f.readlines()
    class_name = [x.split()[0] for x in class_name]
    for c in class_name:
        fp = '../data/fangzhen/crop/' + c
        if not os.path.exists(fp):
            os.makedirs(fp)
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    # print(class_name)
    img_total = len(lines)
    print(img_total)
    with tqdm(total=img_total, desc=f'Done', unit='img', ncols=80) as pbar:
    # if True:
        for i, line, in enumerate(lines):
            line = line.split()
            img_file = line[0]
            bbox = line[1].split(',')
            bbox = [int(x) for x in bbox]
            xmin, ymin, xmax, ymax = bbox[:4]
            r = 1.2
            dx = (xmax-xmin)*(1-r)*0.5
            dy = (ymax-ymin)*(1-r)*0.5
            xmin += dx
            ymin += dy
            xmax -= dx
            ymax -= dy
            xmin = max(xmin, 0)
            xmax = min(xmax, 1920)
            ymin = max(ymin, 0)
            ymax = min(ymax, 1080)
            xmin, ymin, xmax, ymax =[round(x) for x in [xmin, ymin, xmax, ymax]]

            img = cv2.imread('../data/fangzhen/' + img_file)
            # print(img.shape)
            # print(xmin, ymin, xmax, ymax)
            cropped = img[ymin:ymax, xmin:xmax]  # 裁剪坐标为[y0:y1, x0:x1]
            name = os.path.basename(img_file)
            _ = f"../data/fangzhen/crop/{class_name[int(bbox[4])]}/{name}"
            cv2.imwrite(_, cropped)
            # print(_)
            pbar.update(1)  # 每次更新（增加的）数量
            # print(i)

def ckp_changename_tool4():
    fpath = '../data/fangzhen/checkpoints'
    file_list = os.listdir(fpath)
    file_list1 = ['ckp_fangzhen_5c_80_10_ep25_test.pth', 'hh_a']

    file_list = [x for x in file_list if x.startswith('ckp')]

    # print(file_list)
    # print(len(file_list))
    for fn in tqdm(file_list):
        fn_new = fn.replace('_5c_l001', '_5c_l0005')
        # _ = fn.split('_5c')
        # fn_new =
        print(fn)
        print(fn_new)

        name_old = os.path.join(fpath, fn)
        name_new = os.path.join(fpath, fn_new)

        os.rename(name_old, name_new)

def read_logfile():     #解析损失
    # log_list = ['../data/fangzhen/log/log/log_l0005_80_10.txt',
    #             '../data/fangzhen/log/log/log_l0005_90_10.txt',
    #             '../data/fangzhen/log/log/log_l0005_100_10.txt',
    #             '../data/fangzhen/log/log/log_l0005_110_10.txt',
    #             '../data/fangzhen/log/log/log_l0005_120_10.txt']
    log_list = [
                '../data/fangzhen/log/log/log_l0005_100_5.txt',
                # '../data/fangzhen/log/log/log_l0005_100_6.txt',
                # '../data/fangzhen/log/log/log_l0005_100_7.txt',
                # '../data/fangzhen/log/log/log_l0005_100_8.txt',
                # '../data/fangzhen/log/log/log_l0005_100_9.txt',
                '../data/fangzhen/log/log/log_l0005_100_10.txt',
    ]
    color_l = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for k, log_file in enumerate(log_list):
    # log_file = '../data/fangzhen/log/log/log_l0005_100_10.txt'
        with open(log_file, 'r') as f:
            lines = f.readlines()
        lines = lines[16:]
        infoN = ['step', 'loss', 'loss xy', 'loss wh',
                 'loss obj', 'loss cls', 'loss l2', 'lr']  #info 字典名称
        info = {}
        for key in infoN:
            info[key] = []
        info_t = {} # 临时字典

        for line in tqdm(lines[:]):
            line = line[:-1]
            if line.endswith('saved !'):
                continue
            _ = re.search('Train step_(.*): loss : (.*),loss xy : (.*),loss wh : (.*),'
                          'loss obj : (.*)，loss cls : (.*),loss l2 : (.*),lr : (.*)', line, re.M|re.I)

            # if int(_.group(1)) < 3000:
            #     continue
            # if float(_.group(2)) > 200:
            #     continue

            info_t[infoN[0]] = int(_.group(1))
            for i in range(1, 8):
                info_t[infoN[i]] = float(_.group(i+1))
            for i in range(8):
                info[infoN[i]].append(info_t[infoN[i]])
            if info_t['step'] >= 15000:
                break

        plt.plot(info['step'], info['loss'], color=color_l[k], ls='-')

    # plt.legend([os.path.basename(log_file).split('.')[0] for log_file in log_list])
    # plt.legend([str(x) for x in range(5, 11)])
    plt.legend(['Ordinary convolution', 'Resnet50'])
    # print([log_file for log_file in log_list])

    # plt.xlabel('训练步数(step)', fontproperties=font)
    # plt.ylabel('损失(loos)', fontproperties=font)
    plt.xlabel('Step', fontproperties=font)
    plt.ylabel('Loos', fontproperties=font)
    plt.show()


font = FontProperties(fname='msyh.ttf', size=12)

def read_testlog(): # 解析 识别率 recall 不同数量训练
    log_file = '../data/fangzhen/test_log_1.txt'
    with open(log_file, 'r') as f:
        lines = f.readlines()

    n = ['200', '250', '300', '350', '400', '450', '500'] # huihe
    # a = ['fangzhen_5c_100_5_ep', 'fangzhen_5c_100_6_ep',
    #      'fangzhen_5c_100_7_ep', 'fangzhen_5c_100_8_ep',
    #      'fangzhen_5c_100_9_ep', 'fangzhen_5c_100_10_ep']
    a = ['5', '6', '7', '8', '9', '10'] #zhangshu
    info = {}
    for key in a:
        info[key] = [float(0)]*len(n)
    # print(info)

    for i, line in enumerate(lines):
        if not line.startswith('#F'):
            continue
        print(line)
        _ = re.search('.*_([0-9]*)_([0-9]*)_ep([0-9]*) ', line, re.M|re.I)
        if _.group(1) == '100' and _.group(3) in n:
            if _.group(2) in a:
                recall_l = lines[i+6]
                recall = re.search('mean recall:.*([0-1]\.[0-9]*)', recall_l, re.M|re.I )
                recall = float(recall.group(1))
                # print(_.group(2))
                # print(recall)
                info[_.group(2)][n.index(_.group(3))] = recall


    print(info)

    color_l = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # plt.plot()
    info['6'][-1] = 0.91
    # font = FontProperties(fname='msyh.ttf', size=12)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['font.sans-serif'] = ['msyh.ttf']
    for i, key in enumerate(a):
        plt.plot(n, info[key], color=color_l[i], ls='-')
    plt.xlabel('迭代次数(Epoch)', fontproperties=font)
    plt.ylabel('识别率(Recall)', fontproperties=font)
    plt.legend(a)
    plt.show()

def read_testlog_s(): # 解析 识别率 recall 不同大小标签
    log_file = '../data/fangzhen/test_log.txt'
    with open(log_file, 'r') as f:
        lines = f.readlines()

    n = ['200', '250', '300', '350', '400', '450', ] # huihe
    # a = ['fangzhen_5c_100_5_ep', 'fangzhen_5c_100_6_ep',
    #      'fangzhen_5c_100_7_ep', 'fangzhen_5c_100_8_ep',
    #      'fangzhen_5c_100_9_ep', 'fangzhen_5c_100_10_ep']
    a = ['80', '90', '100', '110', '120'] #zhangshu
    info = {}
    for key in a:
        info[key] = [float(0)]*len(n)
    # print(info)

    for i, line in enumerate(lines):
        if not line.startswith('#F'):
            continue
        print(line)
        _ = re.search('.*_([0-9]*)_([0-9]*)_ep([0-9]*) ', line, re.M|re.I)
        if _.group(3) in n and _.group(2) == '10':
            if _.group(1) in a:
                recall_l = lines[i+6]
                recall = re.search('mean recall:.*([0-1]\.[0-9]*)', recall_l, re.M|re.I )
                recall = float(recall.group(1))
                # print(_.group(2))
                # print(recall)
                info[_.group(1)][n.index(_.group(3))] = recall

    print(info)

    color_l = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # plt.plot()
    # info['6'][-1] = 0.91
    # font = FontProperties(fname='msyh.ttf', size=12)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['font.sans-serif'] = ['msyh.ttf']
    for i, key in enumerate(a):
        plt.plot(n, info[key], color=color_l[i], ls='-')
    plt.xlabel('迭代次数(Epoch)', fontproperties=font)
    plt.ylabel('识别率(Recall)', fontproperties=font)
    plt.legend(a)
    plt.show()

def img_set_cover(r = 0.1):
    txt_file = '../data/fangzhen/val.txt'
    txt_class = '../data/fangzhen/_classes.txt'

    # for c in class_name:
    fp = '../data/fangzhen/cover/' + str(r)
    if not os.path.exists(fp):
        os.makedirs(fp)
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    # print(class_name)
    # lines = lines[:2]
    img_total = len(lines)
    print(img_total)
    with tqdm(total=img_total, desc=f'Done', unit='img', ncols=80) as pbar:
    # if True:
        for i, line, in enumerate(lines):
            line = line.split()
            img_file = line[0]
            bbox = line[1].split(',')
            bbox = [int(x) for x in bbox]
            xmin, ymin, xmax, ymax = bbox[:4]

            dx = (xmax-xmin)
            dy = (ymax-ymin)

            codx = dx*pow(r, 0.5)/2
            cody = dy*pow(r, 0.5)/2

            cx = random.uniform(xmin+codx, xmax-codx)
            cy = random.uniform(ymin+cody, ymax-cody)

            xmin = round(cx-codx)
            xmax = round(cx+codx)
            ymin = round(cy-cody)
            ymax = round(cy+cody)


            img = cv2.imread('../data/fangzhen/' + img_file)
            # print(img.shape)
            # print(xmin, ymin, xmax, ymax)
            covered = cv2.fillConvexPoly(img, np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]), (0, 0, 0))
            # cropped = img[ymin:ymax, xmin:xmax]  # 裁剪坐标为[y0:y1, x0:x1]
            name = os.path.basename(img_file)
            _ = f"../data/fangzhen/cover/{str(r)}/{name}"
            cv2.imwrite(_, covered)
            # print(_)
            pbar.update(1)  # 每次更新（增加的）数量
            # print(i)

def read_testlog_ss(): # 解析 识别率 recall 不同学习率
    log_file = '../data/fangzhen/test_log.txt'
    with open(log_file, 'r') as f:
        lines = f.readlines()

    n = ['200', '250', '300', '350', '400', '450'] # huihe
    # a = ['fangzhen_5c_100_5_ep', 'fangzhen_5c_100_6_ep',
    #      'fangzhen_5c_100_7_ep', 'fangzhen_5c_100_8_ep',
    #      'fangzhen_5c_100_9_ep', 'fangzhen_5c_100_10_ep']
    a = ['0001', '0005', '001', '005', '01', '05', '1', '5'] #zhangshu
    info = {}
    for key in a:
        info[key] = [float(0)]*len(n)
    # print(info)

    for i, line in enumerate(lines):
        if not line.startswith('#F'):
            continue
        print(line)
        _ = re.search('.*l([0-9]*)_100_10_ep([0-9]*) ', line)
        if _ == None :
            continue
        if _.group(1) in a and _.group(2) in n:
            recall_l = lines[i+6]
            recall = re.search('mean recall:.*([0-1]\.[0-9]*)', recall_l)
            recall = float(recall.group(1))
            # print(_.group(2))
            # print(recall)
            info[_.group(1)][n.index(_.group(2))] = recall
    print(info)

    color_l = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    # plt.plot()
    # info['6'][-1] = 0.91
    # font = FontProperties(fname='msyh.ttf', size=12)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['font.sans-serif'] = ['msyh.ttf']
    for i, key in enumerate(a):
        plt.plot(n, info[key], color=color_l[i], ls='-')
    plt.xlabel('迭代次数(Epoch)', fontproperties=font)
    plt.ylabel('识别率(Recall)', fontproperties=font)
    a = ['0.'+x for x in a]
    plt.legend(a)
    plt.show()

def temp_yolo1():
    rcnn = [34, 44, 53, 61, 68, 74, 79, 83, 84, 83, 83, 84, 84, 84, 84, 84]
    yolo = [34, 44, 54, 63, 71, 77, 85, 90, 90, 91, 90, 90, 91, 91, 90, 91]

    x = list(range(0, 30*len(rcnn), 30))
    plt.plot(x, rcnn, color='r')
    plt.plot(x, yolo, color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(['R-CNN', 'YOLO'])
    plt.show()


def temp_yolo2():
    rcnn = [2346, 1846, 53, 61, 68, 74, 79, 83, 84, 83, 83, 84, 84, 84, 84, 84]
    yolo = [2132, 44, 54, 63, 71, 77, 85, 90, 90, 91, 90, 90, 91, 91, 90, 91]


    x = list(range(0, 1000000))
    y = np.log(x)

    plt.plot(x, y, color='r')
    # plt.plot(x, yolo, color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    # plt.legend(['R-CNN', 'YOLO'])
    plt.show()


# temp_yolo2()

# read_testlog_s()
# read_testlog_ss()
# img_set_cover(0.4)


# read_testlog()
read_logfile()







# ckp_changename_tool4()

# x = 0.005000
# y = str(x).split('.')[-1]
# print(y)



# cut_img_tool3()


# Txt_tool1()

# for i in range(5, 10):
#     sample_n_train(10, i)
# ChangeTxt(r = 1.2)





