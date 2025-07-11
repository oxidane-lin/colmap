import cv2
import os
import re
import piexif
from PIL import Image
from datetime import datetime
import numpy as np
import argparse

# ---------- 配置 ----------
jpg_quality = 95
min_feature_matches = 50       # 至少匹配点数（用以判断是否足够运动）
min_translation_thresh = 100.0   # 最小平移量阈值（像素）
min_rotation_thresh = 5.0      # 最小旋转角度阈值（角度）
# -------------------------

def parse_srt(srt_path):
    srt_data = {}

    if (not os.path.exists(srt_path)):
        print(f"[W] No srt file provided! {srt_path}")
        return srt_data

    with open(srt_path, "r", encoding="utf-8") as f:
        entries = f.read().split('\n\n')
        for entry in entries:
            lines = entry.strip().splitlines()
            if len(lines) < 3:
                continue
            index = int(lines[0].strip())
            content = ' '.join(lines[2:])
            match_time = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", content)
            if not match_time:
                continue
            dt = datetime.strptime(match_time.group(0), "%Y-%m-%d %H:%M:%S.%f")
            lat_match = re.search(r"latitude:\s*([-\d.]+)", content)
            lon_match = re.search(r"longitude:\s*([-\d.]+)", content)
            rel_alt_match = re.search(r"rel_alt:\s*([-\d.]+)", content)

            srt_data[index] = {
                "time_str": dt.strftime("%Y:%m:%d %H:%M:%S"),
                "latitude": float(lat_match.group(1)) if lat_match else None,
                "longitude": float(lon_match.group(1)) if lon_match else None,
                "rel_alt": float(rel_alt_match.group(1)) if rel_alt_match else None,
            }
    return srt_data

def deg_to_dms_rational(deg_float):
    deg = int(deg_float)
    min_float = (deg_float - deg) * 60
    minute = int(min_float)
    sec_float = (min_float - minute) * 60
    return [(deg, 1), (minute, 1), (int(sec_float * 10000), 10000)]

def write_exif(image_path, meta):
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["0th"][piexif.ImageIFD.DateTime] = meta["time_str"]
    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = meta["time_str"]
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = meta["time_str"]
    if meta["latitude"] is not None and meta["longitude"] is not None:
        lat_ref = "N" if meta["latitude"] >= 0 else "S"
        lon_ref = "E" if meta["longitude"] >= 0 else "W"
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = lat_ref
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = deg_to_dms_rational(abs(meta["latitude"]))
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = lon_ref
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = deg_to_dms_rational(abs(meta["longitude"]))
    if meta["rel_alt"] is not None:
        exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 0
        exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (int(meta["rel_alt"] * 100), 100)

    exif_bytes = piexif.dump(exif_dict)
    img = Image.open(image_path)
    img.save(image_path, exif=exif_bytes)

def is_significant_motion(img1, img2):
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < min_feature_matches:
        return False

    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 仿射估计
    m, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
    if m is None:
        return False

    dx, dy = m[0, 2], m[1, 2]
    translation = np.sqrt(dx**2 + dy**2)
    rotation_deg = np.arctan2(m[1, 0], m[0, 0]) * 180 / np.pi

    return translation > min_translation_thresh or abs(rotation_deg) > min_rotation_thresh

def process_video(video_path, srt_path, output_dir, start_idx, jpg_quality):
    srt_data = parse_srt(srt_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = start_idx
    prev_gray = None
    print(f"Start frame idx for video {video_path}: {frame_idx}")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            significant = True
        else:
            significant = is_significant_motion(prev_gray, gray)

        if significant:
            filename = f"{frame_idx:06d}.JPG"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            meta = srt_data.get(frame_idx - start_idx + 1)
            if meta:
                write_exif(filepath, meta)
            print(f"[{frame_idx}] Saved {filename} (keyframe)")
            prev_gray = gray

        frame_idx += 1

    print(f"Start frame idx for next video: {frame_idx}")
    cap.release()

def main():
    parser = argparse.ArgumentParser(
        description="Split video into images"
    )
    parser.add_argument("--data_path", help="path to *.MP4 and *.SRT")
    parser.add_argument("--video_name",help="name without subfix, for example: DJI_0285")
    parser.add_argument("--start_idx", default= 0, help="start number for image ids")
    parser.add_argument("--subfix", default=".MP4", choices=[".MP4", ".mp4", ".LRF"], help="video subfix")

    args = parser.parse_args()

    video_path = os.path.join(args.data_path, args.video_name + args.subfix)
    srt_path = os.path.join(args.data_path, args.video_name + ".SRT")
    output_dir = os.path.join(args.data_path, args.video_name)
    
    os.makedirs(output_dir, exist_ok=True)

    process_video(video_path, srt_path, output_dir, int(args.start_idx), jpg_quality)

if __name__ == "__main__":
    main()




