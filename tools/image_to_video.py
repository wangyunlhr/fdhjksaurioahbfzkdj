import cv2
import numpy as np
import os
from glob import glob

def images_to_video(folders, output_path, fps=30, layout="vertical"):
    """
    将多个文件夹中相同文件名的图片读取并拼接生成视频。
    
    Args:
        folders: List[str], 包含图片的文件夹路径。
        output_path: str, 输出视频的文件路径。
        fps: int, 视频帧率。
        layout: str, 拼接方式，"horizontal" 或 "vertical"。
    """
    # 确保至少有一个文件夹
    if len(folders) < 2:
        raise ValueError("至少需要两个文件夹进行拼接。")
    
    # 获取第一个文件夹的所有图片文件名（按排序）
    reference_folder = folders[0]
    image_files = sorted(glob(os.path.join(reference_folder, "*.png")) + glob(os.path.join(reference_folder, "*.jpg")))
    if not image_files:
        raise ValueError(f"文件夹 {reference_folder} 中没有找到图片。")
    
    # 提取文件名（不带路径）
    filenames = [os.path.basename(file) for file in image_files]
    
    # 检查每个文件夹是否包含相同的文件名
    for folder in folders[1:]:
        for filename in filenames:
            if not os.path.exists(os.path.join(folder, filename)):
                raise ValueError(f"文件夹 {folder} 中缺少图片文件 {filename}。")
    
    # 读取第一张图片，确定大小
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        raise ValueError("无法读取第一张图片，请检查路径或文件格式。")
    height, width, _ = first_image.shape
    
    # 确定输出帧的大小
    if layout == "horizontal":
        frame_width = width * len(folders)
        frame_height = height
    elif layout == "vertical":
        frame_width = width
        frame_height = height * len(folders)
    else:
        raise ValueError("不支持的拼接方式。请选择 'horizontal' 或 'vertical'。")
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 遍历图片文件
    for filename in filenames:
        images = []
        for folder in folders:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 图片 {img_path} 无法读取，跳过该帧。")
                continue
            images.append(img)
        
        if len(images) != len(folders):
            print(f"警告: 文件 {filename} 缺少部分图片，跳过该帧。")
            continue
        
        # 拼接图片
        if layout == "horizontal":
            frame = np.hstack(images)
        elif layout == "vertical":
            frame = np.vstack(images)
        
        # 写入视频帧
        video_writer.write(frame)
    
    video_writer.release()
    print(f"视频保存到: {output_path}")

# 示例使用
folders = [
    "logs/imgs_multi_box",
    # "logs/imgs_mutli_box_motion",
    "logs/imgs_one_box"
]
output_video_path = "compare_multi_one.mp4"

images_to_video(folders, output_video_path, fps=5, layout="horizontal")
