import os
import glob
import random
import numpy as np

from collections import namedtuple, defaultdict
from read_write_model import read_model, write_model
from read_write_model import Point3D, Image, Camera

def downsample_points(merged_points, voxel_size=0.5, sample_ratio=0.3):
    """
    合并并对点云进行体素网格降采样，使点分布尽量均匀
    :param merged_points: List[Dict[int, Point3D]]
    :param voxel_size: float, 体素边长
    :param max_points_per_voxel: 每个体素最多保留的点数
    :return: List[Point3D]
    """

    all_points = []

    # 取出每个字典的所有值
    for _, point in merged_points.items():
        all_points.append(point)
    print(f"total points: {len(all_points)}")
    if not all_points:
        return {}, {}

    # 提取所有点的xyz坐标
    xyzs = np.array([p.xyz for p in all_points])
    voxel_indices = np.floor(xyzs / voxel_size).astype(int)

    # 分配到voxel
    voxel_dict = defaultdict(list)
    for idx, voxel in enumerate(voxel_indices):
        key = tuple(voxel)
        voxel_dict[key].append(all_points[idx])

    sampled_points = {}
    points_id_remap = {}

    new_id = 0
    for voxel_key, points in voxel_dict.items():

        sample_count = max(1, int(len(points) * sample_ratio))
        selected = random.sample(points, sample_count)

        for p in selected:
            # 更新 id，其他属性保留
            sampled_points[new_id] =Point3D(
                id=new_id,
                xyz=p.xyz,
                rgb=p.rgb,
                error=p.error,
                image_ids=p.image_ids,
                point2D_idxs=p.point2D_idxs
            )
            points_id_remap[p.id] = new_id
            new_id += 1

    return sampled_points, points_id_remap

def add_if_far_enough_copy(a, b, c):
    """返回新列表，不修改原列表"""
    if all(abs(a - x) > c for x in b):
        return True
    return False

def merge_colmap_models(model_dirs, output_dir, downsample=True, voxel_size=0.5, sample_ratio=0.3):
    os.makedirs(output_dir, exist_ok=True)

    # ID 偏移量初始化
    image_id_offset = 1 # 图片也采样了，序号从0计数的
    point3D_id_offset = 0

    merged_cameras = {}
    merged_images = {}
    merged_points3D = {}

    for model_path in model_dirs:
        cameras, images, points3D = read_model(model_path, ".bin")

        for idx, camera in cameras.items():
            merged_cameras[idx] = camera

        cam_framenum = {'center_camera_fov120': [],
                        'left_front_camera': [],
                        'left_rear_camera': [],
                        'right_front_camera': [],
                        'right_rear_camera': [],
                        'rear_camera': [],
                        'center_camera_fov30': [],
                        }

        # 先创建新旧image id的映射关系
        img_id_map = {}
        new_id_counter = 0
        for old_img_id, image in images.items():
            # 采样图片
            if 'center_camera' in image.name:
                continue
            framenum = int(image.name.split('/')[-1].split('.')[0])
            chosenFlag = add_if_far_enough_copy(framenum, cam_framenum[image.name.split('/')[0]], 500)
            if chosenFlag: 
                cam_framenum[image.name.split('/')[0]].append(framenum)

                oripath = os.path.join(model_path, "../../images/",image.name)
                orimaskpath = os.path.join(model_path, "../../mask_combine/",image.name)
                orimaskpath = orimaskpath.replace('.jpg','.png')
                tarpath = os.path.join(output_dir, "rectified/images/", image.name)
                tarmaskpath = os.path.join(output_dir, "rectified/masks/", image.name)
                tarmaskpath = tarmaskpath.replace('.jpg','.png')
                os.makedirs(os.path.dirname(tarpath),exist_ok=True)
                os.makedirs(os.path.dirname(tarmaskpath),exist_ok=True)
                os.system('cp {} {}'.format(oripath,tarpath))
                os.system('cp {} {}'.format(orimaskpath,tarmaskpath))
                print('copyed: {}'.format(oripath.split('/')[-1]))

                new_img_id = new_id_counter + image_id_offset
                img_id_map[old_img_id] = new_img_id
                new_id_counter += 1

        # 再创建新旧point id的映射关系
        point_id_map = {}
        for old_pt_id, pt in points3D.items():
            new_pt_id = old_pt_id + point3D_id_offset
            point_id_map[old_pt_id] = new_pt_id

        for old_img_id, img in images.items():
            if not old_img_id in img_id_map.keys():
                continue
            new_img_id = img_id_map[old_img_id]
            new_img = Image(
                name=img.name,
                id=new_img_id,
                camera_id=img.camera_id,
                tvec=img.tvec,
                qvec=img.qvec,
                xys=img.xys,
                point3D_ids=np.array([-1 if point3d_id == -1 else point_id_map[point3d_id] for point3d_id in img.point3D_ids])
            )
            merged_images[new_img_id] = new_img

        for old_pt_id, pt in points3D.items():
            new_pt_id = point_id_map[old_pt_id]
            new_image_ids = []
            for image_id in pt.image_ids:
                if image_id in img_id_map:
                    new_image_ids.append(img_id_map[image_id])
            new_pt = Point3D(
                id=new_pt_id,
                xyz=pt.xyz,
                rgb=pt.rgb,
                error=pt.error,
                image_ids=np.array(new_image_ids),
                point2D_idxs=pt.point2D_idxs
            )
            merged_points3D[new_pt_id] = new_pt

        image_id_offset = max(merged_images.keys()) + 1
        point3D_id_offset = max(merged_points3D.keys()) + 1

    print(f"merged_cameras: {len(merged_cameras.keys())}")
    print(f"merged_images: {len(merged_images.keys())}")
    print(f"merged_points3D: {len(merged_points3D.keys())}")

    if downsample:
        merged_points3D, points_id_remap = downsample_points(merged_points3D, 0.5, 0.1)
        print(f"downsample merged_points3D: {len(merged_points3D)}")
        sampled_images = {}
        for image_id, img in merged_images.items():
            downsample_ids = []
            for point3D_id in img.point3D_ids.tolist():                 # merged id
                if point3D_id in points_id_remap.keys():                # 被保留下来
                    downsample_ids.append(points_id_remap[point3D_id])  # 存储新id
                else:
                    downsample_ids.append(-1)                           # 该点被删除，存 -1
            new_img = Image(
                name=img.name,
                id=img.id,
                camera_id=img.camera_id,
                tvec=img.tvec,
                qvec=img.qvec,
                xys=img.xys,
                point3D_ids=np.array(downsample_ids)
            )
            sampled_images[new_img.id] = new_img
        merged_images = sampled_images

    # 写入合并后的结果
    merge_model_path = os.path.join(output_dir, "rectified", "sparse", "0") 
    os.makedirs(merge_model_path,exist_ok=True)
    write_model(merged_cameras, merged_images, merged_points3D, merge_model_path)

    print(f"Merged {len(model_dirs)} models into {output_dir}")

if __name__ == "__main__":
    casedic = {} 
    lines = open('/mnt/afs/taoye/code/3_3DGS/chenjianguo/caselist.txt','r').readlines()
    for idx, line in enumerate(lines):
        casedic[idx+1] = line.strip()
    # case1_list = [9,10,28,54,55,7,8,26,74,27,25]
    case1_list = [9,10]
    dataroot = '/mnt/afs/taoye/code/3_3DGS/chenjianguo/data/WAIC_ALL_SEGv2'
    outdataroot = '/mnt/afs/gengmenglin/data/anting_old_2/camera_calibration/'

    camera_subname = '3dgs_format/colmap/sparse_sfm_enu/cameras.bin'
    images_subname = '3dgs_format/colmap/sparse_sfm_enu/images.bin'
    point3D_subname = '3dgs_format/colmap/sparse_sfm_enu/points3D.bin'
    imagepath_subname = '3dgs_format/images'
    sparse_subpath = '3dgs_format/colmap/sparse_sfm_enu'

    os.makedirs(outdataroot,exist_ok=True)

    model_dirs = []
    for caseidx in case1_list:
        camera_absname = os.path.join(dataroot,casedic[caseidx],camera_subname)
        images_absname = os.path.join(dataroot,casedic[caseidx],images_subname)
        point3D_absname = os.path.join(dataroot,casedic[caseidx],point3D_subname)
        imagepath_absname = os.path.join(dataroot,casedic[caseidx],imagepath_subname)
        sparse_abspath = os.path.join(dataroot,casedic[caseidx],sparse_subpath)
        # pdb.set_trace()
        if  os.path.exists(camera_absname) and \
            os.path.exists(images_absname) and \
            os.path.exists(point3D_absname) and \
            os.path.exists(imagepath_absname):

            path = os.path.join(dataroot,casedic[caseidx],"3dgs_format/colmap/sparse_sfm_enu/")
            print(f"to be merged: {path}")
            model_dirs.append(path)

            cameras, images, points3D = read_model(
                path=sparse_abspath, ext='.bin'
            )

    merge_colmap_models(model_dirs, outdataroot)