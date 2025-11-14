import os

def extract_image_names(folder_path):
    # 获取文件夹中所有图片的文件名
    image_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # 创建或打开txt文件
    txt_file_path = os.path.join('/mnt/sdd/yrh/Fei_Bing/code_and_txt_data_process_Fei_Tou', 'image_names_Fei_Tou_4.txt')
    with open(txt_file_path, 'w') as txt_file:
        # 将图片名字写入txt文件，每个名字后面加上" 1"
        for image_name in image_names:
            txt_file.write(f"{image_name} 4\n")

    print(f"图片名字已提取并保存到 {txt_file_path} 文件中。")

# 替换为实际的文件夹路径
folder_path = '/mnt/sdd/yrh/Fei_Bing/dataset_able/Fei_Tou_4'
extract_image_names(folder_path)

# 一共改三个地方，一个是txt文件的名字，一个是标签的值，一个是处理的图片的路径
