import os

# 读取数据函数,返回list类型的数据
# FileName为输入文件的名称
# NewFileName为希望生成的txt文件名称，默认为'transformed_data.txt'
# r = 0.8   缩放比例r默认0.8
def ChangeTxt(FileName, NewFilepath = 'transformed_data.txt', r = 0.8):
    txtname = os.path.basename(FileName)
    txtname = txtname.split('_')
    NewFileName = NewFilepath+txtname[0] + f'_{int(r*100)}_' + txtname[2]
    with open(FileName, 'r') as txtData:
        # 读取数据函数,返回list类型的数据
        lines = txtData.readlines()
    with open(NewFileName, 'w') as txtData_w:
        for line in lines:
            lineData = line.split()  # 分割空白和\n
            # lineData3 = lineData[:]
            Box = lineData[-1].split(',')
            lineData2 = Box[:]
            del Box[-1]
            # print(Box)
            list = []
            for box in Box:
                boxbox = int(box) * r
                list.append(str(int(boxbox)))
            # print(list[0])
            #
            # 如果右下角的坐标超过屏幕最大限制的话，令其相应坐标等于屏幕坐标

            if int(list[0]) < 0:
                list[0] = 0
            if int(list[1]) < 0:
                list[1] = 0

            if int(list[2]) >= 1920:
                list[2] = 1920
            if int(list[3]) >= 1280:
                list[3] = 1280

            newLine = lineData[0] + ' {},{},{},{},{}'.format(list[0], list[1], list[2], list[3], lineData2[-1])
            line = line.replace(line, newLine)
            txtData_w.write(line + '\n')
            print(line)
            # txtData.close()
            # txtData_w.close()



# loadData("train_100_10.txt")



