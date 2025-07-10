import os
import json
import cv2
import glob
import argparse
import numpy as np

def load_camera_params(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    intrinsics = np.array(data['CameraIntrinsics'])
    distortion = np.array(data['Distortion'])
    return intrinsics, distortion

def undistort_image(image, K, dist_coeffs):
    h, w = image.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, K, dist_coeffs, None, new_K)
    return undistorted

def process_folder(json_path, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    K, dist_coeffs = load_camera_params(json_path)
    print(K, dist_coeffs)

    # 支持 PNG 和 JPG 图片
    image_paths = glob.glob(os.path.join(input_folder, '*.png')) + \
                  glob.glob(os.path.join(input_folder, '*.jpg')) + \
                  glob.glob(os.path.join(input_folder, '*.PNG')) + \
                  glob.glob(os.path.join(input_folder, '*.JPG')) + \
                  glob.glob(os.path.join(input_folder, '*.JPEG')) + \
                  glob.glob(os.path.join(input_folder, '*.jpeg'))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片：{img_path}")
            continue
        undistorted = undistort_image(img, K, dist_coeffs)
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, undistorted)
        print(f"保存去畸变图片：{save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='去畸变图像处理脚本')
    parser.add_argument('--json', type=str, required=True, help='包含相机参数的 JSON 文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出图片文件夹路径')
    args = parser.parse_args()
    print(args)

    process_folder(args.json, args.input, args.output)
