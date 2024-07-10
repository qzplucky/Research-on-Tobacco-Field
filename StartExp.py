

# 读取目标yml文件并展示内容
# 展示模型文件内容，以确认已训练模型数量
# 提供手动修改yml文件内train\val部分内容的方法
# 展示文件内容确认修改成功
# 确认修改后写入文件


# 自动化流程
import os

import ruamel.yaml
import codecs

# 实现Python执行命令
import subprocess

from collections import defaultdict


# 使用ruemal.yaml加载yaml文件
yaml = ruamel.yaml.YAML()
filePath = '/home/zj/Orgseg/PaddleSeg/configs/road_seg/deepglobe.yml'


# 用于待训练文件加载
ModelSavePath = './output/final/'
DatasetPath = './data/dataset/'
ExcludeFileName = 'exclude.txt'





def get_user_input(prompt, default_value):
    user_input = input(f"{prompt} (default: {default_value}): ").strip()
    return user_input if user_input else default_value

def display_yaml_content(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            print(" " * indent + f"{key}:")
            display_yaml_content(value, indent + 4)
    elif isinstance(data, list):
        for item in data:
            display_yaml_content(item, indent)
    else:
        print(" " * indent + str(data))

def comd(filename):

    # 打开文件并计算行数
    with codecs.open(DatasetPath+filename + "_train.txt", 'r') as file:
        line_count = sum(1 for line in file)

    save_dir = "/home/zj/Orgseg/PaddleSeg/output/final/" + filename + "_" + str(line_count)
    config = '/home/zj/Orgseg/PaddleSeg/configs/road_seg/pp_liteseg_stdc1_deepglobe_1024x1024_80k.yml'
    # command = "python val.py --model_path /home/zj/PaddleSeg/output/unet_/best_model/model.pdparams --config /home/zj/Orgseg/PaddleSeg/configs/newuet/unet.yml"
    commands = "nohup python train.py --config " + config +"  --do_eval --num_workers 3 --save_interval 1000 --save_dir "+ save_dir + " --use_vdl > /home/zj/Orgseg/PaddleSeg/outputlog/"+ filename +".log 2>&1 &"
    
    # 执行命令
    UserCmd = get_user_input("Start the command:{}?".format(commands),False)
    if UserCmd != False:
        try:
            subprocess.run(commands, check=True,shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Command execution failed: {e}")
    else:
        print("End the train!")


#################################
# 文件加载
#################################

# 列出 ModelSavePath 内的文件/文件夹列表
model_files = os.listdir(ModelSavePath)
# print(f"Files/Folders in {ModelSavePath}:")

model_dict = defaultdict(list)
print("正在加载模型数据")
for file in model_files:
    filename = file.split("_")
    model_dict[filename[0]].append(filename[1])
# 列出 data/dataset 内除了 exclude.txt 以外的所有 .txt 文件

print("正在加载训练文件数据")
dataset_files = [file for file in os.listdir(DatasetPath) if file.endswith('.txt') and file != ExcludeFileName]
wait_train_files = set()
model_dict_key = list(model_dict.keys())

for file in dataset_files:
    filename = file.split("_")
    # print(filename[0],model_dict_key)
    if filename[0] in model_dict_key and filename[1] in model_dict[filename[0]]:
        continue
    else:
        wait_train_files.add(filename[0]+"_"+filename[1])
print("待训练文件如下所示：")
for i in wait_train_files:
    print(i)

selectFilename = get_user_input("请选择待训练文件(输入下标)",'n')



if selectFilename != 'n':
    print("开始加载yaml文件:{}".format(filePath))
    #################################
    # yaml文件加载
    #################################
    wait_train_files = list(wait_train_files)
    with codecs.open(filePath, 'r') as f:
        yml_content = yaml.load(f)

    # 输出yaml内容
    display_yaml_content(yml_content)

    # 2. 提供手动修改yml文件内train\val部分内容的方法
    # 修改 train_dataset 的 dataset_root
    yml_content['train_dataset']['train_path'] = "data/dataset/" + wait_train_files[int(selectFilename)] + "_train.txt"
    # # 修改 val_dataset 的 dataset_root
    yml_content['val_dataset']['val_path'] = "data/dataset/" + wait_train_files[int(selectFilename)] + "_val.txt"
    # # 3. 展示文件内容确认修改成功
    print(f'\n修改后的文件内容:')
    display_yaml_content(yml_content)

    # 4. 确认修改后写入文件
    confirm_save = input("需要保存本次修改吗? (y/n): ").strip().lower()
    if confirm_save == 'y':
        yaml.default_flow_style = False
        with codecs.open(filePath, 'w') as f:
            yaml.dump(yml_content, f)
        print("修改成功！已覆盖保存！进行下一步操作！")
        comd(wait_train_files[int(selectFilename)])
    else:
        print("No changes were saved.")
else:
    print("终止训练！")


