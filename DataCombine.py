import os

def combine_files(folder_path, output_folder):
    # 获取文件夹中所有文件
    files = [f for f in os.listdir(folder_path) if (os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".txt") and f != "exclude.txt")]
    files = sorted(files)
    # 遍历所有文件，进行两两组合
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            file1 = files[i]
            file2 = files[j]

            # 提取通道类型 xx 和 yy
            xx1, yy1 = file1.split("_")[1:3]
            xx2, yy2 = file2.split("_")[1:3]
            head1 = file1.split("_")[0]
            head2 = file2.split("_")[0]
            # 检查通道类型是否匹配及是否是单数据集
            if len(head1)==2 and len(head2)==2 and xx1 == xx2 and yy1 == yy2:
                print("正在组合文件{} + {}".format(file1,file2))
                # 组合文件内容
                with open(os.path.join(folder_path, file1), 'r') as f1, open(os.path.join(folder_path, file2), 'r') as f2:
                    combined_content = f1.read() + f2.read()

                # 构建新文件名，例如 s14_RGB_train.txt
                new_file_name = f"s{head1[1:]}{head2[1:]}_{xx1}_{yy1}"
                new_file_path = os.path.join(output_folder, new_file_name)

                if os.path.exists(new_file_path)==False:
                    # 将组合后的内容写入新文件
                    with open(new_file_path, 'w') as new_file:
                        new_file.write(combined_content)

                    print(f"Combined {file1} and {file2} into {new_file_name}")
                else:
                    print("文件已存在，跳过组合！")
                    continue

# 指定数据集所在的文件夹路径和输出文件夹路径
data_folder = "./data/dataset/"
output_folder = "./data/dataset/"


# 执行组合文件操作
combine_files(data_folder, output_folder)
