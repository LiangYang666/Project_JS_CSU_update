import re
import os
from os import listdir
from os.path import join
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-dir', '--dataset_dir', type=str, help="dataset_dir")
# args = parser.parse_args()
# dataset_dir = args.dataset_dir


dataset_dir = '../data/Chuan_z'

input_text_file = dataset_dir+'/val.txt'
class_text_file = dataset_dir+'/_classes.txt'
output_text_path = dataset_dir+'/txt_file_truth'

if not os.path.exists(output_text_path):
    os.makedirs(output_text_path)
    print("{} Directory created successfully!".format(output_text_path))


if __name__ == '__main__':
    # # 生成train文件
    # out_file = open('train.txt', 'w')
    # for i in range(1, 651):
    #     out_file.write('data/obj/{:0>3d}.jpg'.format(i) + '\n')
    # assert False

    class_names = []
    with open(class_text_file, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip('\n')
        line = line.rstrip()
        class_names.append(line)
    print(class_names)
    f = open(input_text_file, 'r')
    lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n')
        line = line.rstrip()
        if line[-1] is 'g':
            pass
        else:
            get = line.split(' ')
            txt_name = get[0].split('/')[-1]
            with open(output_text_path + '/' + txt_name.split('.')[0] + '.txt', 'w') as fout:
                for index, get_each in enumerate(get[1:]):
                    bb_each = get_each.split(',')
                    each_cls_name = class_names[int(bb_each[-1])]

                    if len(bb_each) == 5:
                        fout.write(each_cls_name+' ' + ' '.join(bb_each[:-1]))
                    elif len(bb_each) == 6:
                        confid = bb_each[-2]
                        fout.write(each_cls_name+' ' + confid + ' ' + ' '.join(bb_each[:-2]))
                    if index < len(get[1:])-1:
                        fout.write('\n')
            # print(get)
            # print('**********************************************')
    f.close()
    print('OK')

# print(num_classes)
