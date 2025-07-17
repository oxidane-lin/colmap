from seperate_video import process_video, process_video_by_idx
import os
import cv2
import concurrent.futures
from pathlib import Path

# 你的输入文件夹
input_dir = "/data/gs/anting/0712_card/cam2_card1/"
output_base = "/path/to/output_dir"

# 获取所有 .lrf 文件
def list_lrf_files_sorted(input_dir):
    lrf_files = [f for f in Path(input_dir).glob("*.MP4")]
    return sorted(lrf_files, key=lambda p: p.name)

# 获取视频帧数
def get_video_frame_count(filepath):
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        print(f"无法打开视频: {filepath}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

# 每个视频处理任务
def process_single_video(lrf_path, start_idx):
    frame_count = get_video_frame_count(lrf_path)
    print(f"{lrf_path.name} 帧数: {frame_count}")

    # 输出路径 = output_base / 视频名（无后缀）
    out_dir = Path(input_dir) / lrf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 调用你的关键帧提取函数
    srt_path = Path(input_dir) / (lrf_path.stem + ".SRT")
    print(f"start_idx = ", start_idx)
    process_video(str(lrf_path), srt_path,  # 如果你有 .srt 匹配逻辑，可自动推导
                  output_dir=str(out_dir),
                  start_idx=start_idx,
                  jpg_quality=95)
    return lrf_path.name

# 每个视频处理任务
def process_single_video_by_idx(mp4_path, start_idx, chosen_index):
    frame_count = get_video_frame_count(mp4_path)
    print(f"{mp4_path.name} 帧数: {frame_count}")

    # 输出路径 = output_base / 视频名（无后缀）
    out_dir = Path(input_dir) / mp4_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 调用你的关键帧提取函数
    srt_path = Path(input_dir) / (mp4_path.stem + ".SRT")
    print(f"start_idx = ", start_idx)
    process_video_by_idx(str(mp4_path), srt_path,  # 如果你有 .srt 匹配逻辑，可自动推导
                  output_dir=str(out_dir),
                  start_idx=start_idx,
                  jpg_quality=95, chosen_index=chosen_index)
    return mp4_path.name

def extract_frame_indices(folder_path):
    """
    提取文件夹中所有.jpg文件的编号，返回一个整数列表。
    """
    frame_list = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".jpg"):
            name, _ = os.path.splitext(file)
            if name.isdigit():
                frame_list.append(int(name))
    frame_list.sort()
    return frame_list

def main():
    lrf_list = list_lrf_files_sorted(input_dir)
    print(f"共发现 {len(lrf_list)} 个 .lrf 文件")
    print(lrf_list)

    frame_cnt, frame_acc = [], []
    start_idx = 0
    for lrf_path in lrf_list:
        frame_count = get_video_frame_count(lrf_path)
        frame_cnt.append(frame_count)
        frame_acc.append(start_idx)
        start_idx += frame_count
    
    print(frame_cnt)
    print(frame_acc)
    print(f"All frames: {start_idx}")

    chosen_index = extract_frame_indices("/data/gs/anting/0712_card/seg5/")

    # 最多并行 5 个任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for lrf_path, start_idx in zip(lrf_list, frame_acc):
            futures.append(executor.submit(process_single_video_by_idx, lrf_path, start_idx, chosen_index))

        # 等待所有任务完成
        for f in concurrent.futures.as_completed(futures):
            print(f"处理完成: {f.result()}")

if __name__ == "__main__":
    main()
