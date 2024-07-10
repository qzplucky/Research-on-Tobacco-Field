import os

def get_user_input(prompt, default_value):
    user_input = input(f"{prompt} (default: {default_value}): ").strip()
    return user_input if user_input else default_value

def check_files(base_folder):
    # 获得所有子文件夹
    subfolders = os.listdir(base_folder)
    check_name = "best_model"
    # 对子文件夹进行便利搜索

    rmList = []
    for i in subfolders:
        subfolder_path = os.path.join(base_folder,i)
        SsubFolder = os.listdir(subfolder_path)
        if check_name not in SsubFolder:
            rmList.append(subfolder_path)
            print(subfolder_path)

    flags = get_user_input("检测到以上可能存在问题文件夹，是否删除以下目录？",False)

    if flags == False:
        print("已终止删除")
    else:
        for i in rmList:
            os.rmdir(i)


if __name__ == "__main__":
    base_folder = "/home/zj/Orgseg/PaddleSeg/output/final"
    check_files(base_folder)
