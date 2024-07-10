import os
import cv2
import numpy as np

# 基本目标：NDVI与 GRVI的计算与保存
# NDVI = (NIR-RED)/(NIR+RED)  NIR和Red通道
# GRVI = (Green-Red)/(Green+Red) Green和Red通道


def calculate_index(nir, red, mode):
    
    # 预处理，将零值替换为1
    nir[nir == 0] = 1
    red[red == 0] = 1 
    denominator = nir + red
    denominator[denominator == 0] = 1
    if mode == 'NDVI':
        index = (nir - red) / denominator
    elif mode == 'GRVI':
        index = (nir - red) / denominator
    else:
        raise ValueError("Invalid mode. Please choose 'NDVI' or 'GRVI'.")
    return index

def process_images(folder1, folder2, output_folder, mode):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    for file1, file2 in zip(files1, files2):
        img1 = cv2.imread(os.path.join(folder1, file1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(folder2, file2), cv2.IMREAD_GRAYSCALE)

        index = calculate_index(img1, img2, mode)
        denominator = np.max(index) - np.min(index)
        denominator = np.where(denominator == 0, 1, denominator)


        # Normalize values to the range [0, 255]
        index = ((index - np.min(index)) / denominator )* 255
        # Convert to 16-bit unsigned integer for saving
        index = index.astype(np.uint16)
        
        if mode == 'NDVI': 
            filename = file2.replace("red", mode)
        elif mode == 'GRVI':
            filename = file2.replace("red", mode)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, index)

if __name__ == "__main__":
# 29_transparent_mosaic_red__0__5
    datasetPath = ["/home/zj/Orgseg/PaddleSeg/data/dataset/green/",'/home/zj/Orgseg/PaddleSeg/data/dataset/nir/']
    OutPutPath = ["/home/zj/Orgseg/PaddleSeg/data/dataset/GRVI/",'/home/zj/Orgseg/PaddleSeg/data/dataset/NDVI/']
    redPath = "/home/zj/Orgseg/PaddleSeg/data/dataset/red"
    mode = ['GRVI','NDVI']
    for i,j,z in zip(datasetPath,mode,OutPutPath):
        print("正在处理模式：{}".format(j))
        process_images(i, redPath, z, j)
