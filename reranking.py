import numpy as np

import cv2
import os

import matplotlib.pyplot as plt

output_path = "C:/Users/YorkeYao/Desktop/ICCV/Baseline/output_tem.txt"

id = "07d77189d739d94b4f916d4d18f00247"

output_img = np.zeros( (200, 200, 100))

with open(output_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

    for line in lines:
        line_split = line.split(" ")
        if line_split[0] == id:
            if int(line_split[1]) < 200 and int(line_split[2]) < 200:
                output_img[ int(line_split[1]), int(line_split[2])] = np.exp ( float(line_split[-1]) / ( int(line_split[2]) - int(line_split[1]) ) ) ** 2

print (output_img.max())


output_img_new = np.zeros( (200, 200, 3))

output_img_new[:,:,0] = output_img

output_img_new[:,:,1] = output_img

output_img_new[:,:,2] = output_img


task3_label_dir = os.path.expanduser("D:\Datasets\LRW-1000\MAVSR2019_val_task3_kws\kws\info")

for num, label_file in enumerate(os.listdir(task3_label_dir)):
    label_file_path = os.path.join(task3_label_dir, label_file)

    label_file_num = int(label_file.strip('.txt'))

    with open(label_file_path, 'r', encoding = 'utf-8') as lf:
        labels_info = lf.readlines()

    labels_info = [item.strip('\n') for item in labels_info]

    video_id = labels_info[0]

    if video_id != id:
        continue

    labels_info.pop(0)
    labels_info.pop(-1)

    while '' in labels_info:
        labels_info.remove('')

    for label in labels_info:
        label_info = label.split(',')

        label_pinyin = label_info[1]

        if int(float(label_info[2])*25)+1 < 200 and int(float(label_info[3])*25)+1 < 200:

            

            print (label_pinyin, int(float(label_info[2])*25)+1 , int(float(label_info[3])*25)+1)

            output_img_new[ int(float(label_info[2])*25)+1, int(float(label_info[3])*25)+1, 0] = 1

            output_img_new[int(float(label_info[2])*25)+1, int(float(label_info[3])*25)+1, 1] = 0

            output_img_new[int(float(label_info[2])*25)+1, int(float(label_info[3])*25)+1, 2] = 0


# src_RGB = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

# print (np.shape (src_RGB))

# imgplot = plt.imshow(output_img)
# plt.show()

imgplot = plt.imshow(output_img_new)
plt.show()
assert(0)
# output_img
# src = cv2.imread("demo.jpg", 0)
# 然后用ctvcolor（）函数，进行图像变换。
src_RGB = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
# 显示图片
# cv2.imshow("input", src)
cv2.imshow("output", src_RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

            # print (line_split[0])
            # assert (0)
    