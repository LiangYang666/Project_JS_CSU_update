from pathlib import Path
import glob
import os

import xml.etree.ElementTree as ET
import random
from tqdm import tqdm

data_dir = '../../data/Chuan_z'
train_path = '../../data/Chuan_z/datasets/5class'
train_txt_f = os.path.join(data_dir, 'train.txt')
class_txt = os.path.join(data_dir, '_classes.txt')

def get_all_file(p):
    p = Path(p)  # os-agnostic
    f = []
    if p.is_dir():  # dir
        f += glob.glob(str(p / '**' / '*.*'), recursive=True)
    return f

def get_boxes_from_xml(file):
    tree = ET.parse(file)  # 获取xml文件
    root = tree.getroot()
    # filename = root.find('filename').text
    # object = root.find('object')
    info = []
    for object in root.findall('object'):
        name = object.find('name').text
        bandbox = object.find('bndbox')
        xmin = int(bandbox.find('xmin').text)
        ymin = int(bandbox.find('ymin').text)
        xmax = int(bandbox.find('xmax').text)
        ymax = int(bandbox.find('ymax').text)
        each = [name, xmin, ymin, xmax, ymax]
        info.append(each)
    return info

# %%
if __name__ == "__main__":
    file_l = get_all_file(train_path)
    print(file_l)

# %%
    file_l = [x.split('Chuan_z/')[1] for x in file_l]
# %%
    image_list = [x for x in file_l if x.endswith('.jpg')]
    xml_list = [x for x in file_l if x.endswith('.xml')]
    assert len(image_list) == len(xml_list)
    txt_dic = {}
    class_names = ['burke', 'freedom', 'nimitz', 'ticonderoga', 'wasp']
    for i in xml_list:
        f = os.path.join(data_dir, i)
        info = get_boxes_from_xml(f)
        info = [[x[0].lower()]+x[1:] for x in info]
        imgf = f[:-3]+'jpg'
        assert Path(imgf).is_file(), f'{imgf} is not exist'
        imgf = imgf.split('Chuan_z/')[1]
        txt_dic[imgf] = info
    # for i in txt_dic.values():
    #     for name_bbox in i:
    #         if name_bbox[0] not in class_names:
    #             class_names.append(name_bbox[0])
    # class_names.sort()
    print(txt_dic)
    print(class_names)  # ['burke', 'freedom', 'nimitz', 'ticonderoga', 'wasp']

# %%
    yolo_txt_list = []
    for key in txt_dic.keys():
        s_line = key
        info = txt_dic[key]
        flag = False
        for box in info:
            name, xmin, ymin, xmax, ymax = box
            if name in class_names:
                s_line += " {},{},{},{},{}".format(xmin, ymin, xmax, ymax, class_names.index(name))
                flag = True
            else:
                raise Exception(f'{key} have a new class that not in our data')
        if flag:  # 如果有目标就加进去
            yolo_txt_list.append(s_line)

    with open(train_txt_f, 'w') as f:
        for line in yolo_txt_list:
            f.write(line+'\n')
    with open(class_txt, 'w') as f:
        for line in class_names:
            f.write(line+'\n')


#%%