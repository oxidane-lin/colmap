import numpy as np
import json

def compute_similarity_transformation(source_points, target_points):
    """
    计算从source_points到target_points的最佳相似变换(s,R,t)
    包括缩放、旋转和平移
    
    参数:
        source_points: (N,3) numpy数组，源点云中的匹配点
        target_points: (N,3) numpy数组，目标点云中的匹配点
        
    返回:
        s: 缩放因子
        R: (3,3)旋转矩阵
        t: (3,)平移向量
        transformation_matrix: (4,4)齐次变换矩阵
    """
    # 确保输入正确
    assert source_points.shape == target_points.shape
    assert source_points.shape[1] == 3
    N = source_points.shape[0]  # 点对数量
    
    # 计算质心
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    
    # 中心化点集
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target
    
    # 计算缩放因子
    # 源点集的"方差"
    var_source = np.sum(np.linalg.norm(centered_source, axis=1)**2)
    
    # 计算协方差矩阵
    H = centered_source.T @ centered_target
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = Vt.T @ U.T
    
    # 处理反射情况(确保是纯旋转)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    # 计算缩放因子
    scale = np.trace(np.diag(S)) / var_source
    
    # 计算平移向量
    t = centroid_target - scale * (R @ centroid_source)
    
    # 构建4x4齐次变换矩阵
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = scale * R
    transformation_matrix[:3, 3] = t
    
    return scale, R, t, transformation_matrix

def apply_transformation(points, transformation_matrix):
    """
    应用齐次变换矩阵到点集上
    
    参数:
        points: (N,3) numpy数组，要变换的点
        transformation_matrix: (4,4)齐次变换矩阵
        
    返回:
        transformed_points: (N,3)变换后的点
    """
    # 转换为齐次坐标
    homogeneous_points = np.column_stack([points, np.ones(points.shape[0])])
    
    # 应用变换
    transformed_homogeneous = (transformation_matrix @ homogeneous_points.T).T
    
    # 转换回3D坐标
    return transformed_homogeneous[:, :3]

def save_transformation_matrix(filename, matrix, description=""):
    """
    保存变换矩阵到文件
    
    参数:
        filename: 保存文件名
        matrix: (4,4)变换矩阵
        description: 可选的描述文本
    """
    with open(filename, 'w') as f:
        if description:
            f.write(f"# {description}\n")
        np.savetxt(f, matrix, fmt='%.8f',delimiter=',')

def save_transformation_json(filename, transformation_matrix, 
                           source_ply="", target_ply="", 
                           description=""):
    """
    将变换矩阵和元数据保存为JSON格式
    
    参数:
        filename: 输出的JSON文件路径
        transformation_matrix: (4,4)变换矩阵
        source_ply: 源点云PLY文件路径(可选)
        target_ply: 目标点云PLY文件路径(可选)
        description: 描述文本(可选)
    """
    # 从变换矩阵中提取参数
    s = np.linalg.norm(transformation_matrix[:3, 0])  # 缩放因子
    R = transformation_matrix[:3, :3] / s  # 旋转矩阵
    t = transformation_matrix[:3, 3]     # 平移向量
    
    # 转换为可JSON序列化的格式
    data = {
        "description": description,
        "source_ply": source_ply,
        "target_ply": target_ply,
        "transform_type": "similarity",  # similarity/rigid/affine

        "parameters": {
            "scale": float(s),
            "rotation": R.tolist(),
            "translation": t.tolist()
        },
        "transformation_matrix": transformation_matrix.tolist()
    }
    
    # 保存为JSON文件
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"变换参数已保存到 {filename}")

import argparse
import open3d
import struct
import collections

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb"]
)

class Visualizer:
    def __init__(self):
        self.cameras = []
        self.images = []
        self.points3D = []
        self.__vis = None

    def read_ply_new(self, path):
        """
        读取PLY格式的点云文件，完整处理ASCII和二进制格式(包括字节序)
        """

        points_3d = {}
        with open(path, 'rb') as f:
            # 检查文件头
            if b'ply' not in f.readline():
                raise ValueError("不是有效的PLY文件")
                
            # 初始化解析变量
            format = 'ascii'
            byte_order = '<'  # 默认小端
            vertex_count = 0
            properties = []
            property_types = []
            
            # 解析文件头
            while True:
                line = f.readline().decode('ascii').strip()
                if line == 'end_header':
                    break
                    
                if line.startswith('format'):
                    parts = line.split()
                    if parts[1] == 'binary_little_endian':
                        format = 'binary'
                        byte_order = '<'
                    elif parts[1] == 'binary_big_endian':
                        format = 'binary'
                        byte_order = '>'
                    elif parts[1] == 'ascii':
                        format = 'ascii'
                        
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                    
                elif line.startswith('property'):
                    parts = line.split()
                    prop_type = parts[1]
                    prop_name = parts[-1]
                    properties.append(prop_name)
                    property_types.append(prop_type)
        
            # 准备数据存储
            has_color = all(c in properties for c in ['red', 'green', 'blue'])
            has_normal = all(n in properties for n in ['nx', 'ny', 'nz'])
            
            points = np.zeros((vertex_count, 3), dtype=np.float32)
            colors = np.zeros((vertex_count, 3), dtype=np.uint8) if has_color else None
            normals = np.zeros((vertex_count, 3), dtype=np.float32) if has_normal else None
            
            # 处理ASCII格式
            if format == 'ascii':
                for i in range(vertex_count):
                    data = f.readline().decode('ascii').split()

                    points[i] = [float(data[properties.index('x')]), 
                                float(data[properties.index('y')]), 
                                float(data[properties.index('z')])]
                    
                    if has_normal:
                        normals[i] = [float(data[properties.index('nx')]),
                                    float(data[properties.index('ny')]),
                                    float(data[properties.index('nz')])]
                        
                    if has_color:
                        colors[i] = [int(data[properties.index('red')]),
                                    int(data[properties.index('green')]),
                                    int(data[properties.index('blue')])]
                    
                    points_3d[i] = Point3D(
                                id=i,
                                xyz=points[i],
                                rgb=colors[i] if has_color else [255, 255, 255]
                            )
            
            # 处理二进制格式
            else:
                # 构建结构体格式字符串
                struct_fmt = byte_order
                type_map = {
                    'char': 'b', 'uchar': 'B',
                    'short': 'h', 'ushort': 'H',
                    'int': 'i', 'uint': 'I',
                    'float': 'f', 'double': 'd'
                }
                
                for prop_type in property_types:
                    struct_fmt += type_map.get(prop_type, 'f')  # 默认float
                
                # 计算每个顶点的大小
                vertex_size = struct.calcsize(struct_fmt)
                
                # 读取顶点数据
                for i in range(vertex_count):
                    vertex_data = f.read(vertex_size)
                    data = struct.unpack(struct_fmt, vertex_data)
                    
                    # 按属性名提取数据
                    points[i] = [data[properties.index('x')],
                                data[properties.index('y')],
                                data[properties.index('z')]]
                    
                    if has_normal:
                        normals[i] = [data[properties.index('nx')],
                                    data[properties.index('ny')],
                                    data[properties.index('nz')]]
                    
                    if has_color:
                        # 处理颜色可能是uchar(0-255)或float(0-1)的情况
                        red = data[properties.index('red')]
                        green = data[properties.index('green')]
                        blue = data[properties.index('blue')]
                        
                        if property_types[properties.index('red')] == 'float':
                            colors[i] = [int(red*255), int(green*255), int(blue*255)]
                        else:
                            colors[i] = [red, green, blue]
                    points_3d[i] = Point3D(
                                id=i,
                                xyz=points[i],
                                rgb=colors[i] if has_color else [255, 255, 255]
                            )
        return points_3d

    def read_ply(self, path):
        """
        读取PLY格式的点云文件
        
        参数:
            path: PLY文件路径
            
        返回:
            points: (N,3) numpy数组，包含点云坐标(x,y,z)
            colors: (N,3) numpy数组，包含点云颜色(r,g,b)，如果没有颜色则为None
            normals: (N,3) numpy数组，包含点云法向量(nx,ny,nz)，如果没有法向量则为None
        """
        with open(path, 'rb') as f:
            # 检查文件头
            if b'ply' not in f.readline():
                raise ValueError("不是有效的PLY文件")
                
            # 读取文件头信息
            format = 'ascii'
            vertex_count = 0
            properties = []
            while True:
                line = f.readline().decode('ascii').strip()
                if line == 'end_header':
                    break
                    
                if line.startswith('format'):
                    parts = line.split()
                    if parts[1] == 'binary_little_endian':
                        format = 'binary'
                        
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                    
                elif line.startswith('property'):
                    properties.append(line.split()[-1])
        
            # 准备数据存储
            has_color = 'red' in properties and 'green' in properties and 'blue' in properties
            has_normal = 'nx' in properties and 'ny' in properties and 'nz' in properties
            
            points = np.zeros((vertex_count, 3), dtype=np.float32)
            colors = np.zeros((vertex_count, 3), dtype=np.uint8) if has_color else None
            normals = np.zeros((vertex_count, 3), dtype=np.float32) if has_normal else None
            
            points_3d = {}
            # 读取顶点数据
            if format == 'ascii':
                for i in range(vertex_count):
                    data = f.readline().decode('ascii').split()
                    idx = 0
                    points[i] = [float(data[idx]), float(data[idx+1]), float(data[idx+2])]
                    idx += 3
                    
                    if has_normal:
                        normals[i] = [float(data[idx]), float(data[idx+1]), float(data[idx+2])]
                        idx += 3
                        
                    if has_color:
                        # if 'uchar' in properties:  # 颜色是0-255的整数
                        if True:
                            colors[i] = [int(data[idx]), int(data[idx+1]), int(data[idx+2])]
                            if (np.all(colors[i] == [255, 255, 255])):
                                colors[i] = [0, 0, 0]
                        else:  # 颜色是0-1的浮点数
                            colors[i] = [float(data[idx])*255, float(data[idx+1])*255, float(data[idx+2])*255]
                        
                    
                    points_3d[i] = Point3D(
                        id=i,
                        xyz=points[i],
                        rgb=colors[i] if has_color else [255, 255, 255]
                    )

            else:  # 二进制格式
                # 确定每个顶点的字节大小和格式
                fmt = '>'  # 小端字节序
                property_formats = []
                for prop in properties:
                    if prop in ['x', 'y', 'z', 'nx', 'ny', 'nz']:
                        fmt += 'd'  # double
                        property_formats.append('d')
                    elif prop in ['red', 'green', 'blue']:
                        fmt += 'B'  # uint8
                        property_formats.append('B')
                    else:
                        fmt += 'd'  # 其他属性默认按double处理
                        property_formats.append('d')
                vertex_size = struct.calcsize(fmt)
                
      
                for i in range(vertex_count):
                    vertex_data = f.read(vertex_size)
                    data = struct.unpack(fmt, vertex_data)
                    
                    # 解析数据
                    x_idx = properties.index('x')
                    points[i] = [data[x_idx], data[x_idx+1], data[x_idx+2]]
                    
                    if has_normal:
                        nx_idx = properties.index('nx')
                        normals[i] = [data[nx_idx], data[nx_idx+1], data[nx_idx+2]]
                    
                    if has_color:
                        r_idx = properties.index('red')
                        colors[i] = [data[r_idx], data[r_idx+1], data[r_idx+2]]
                        # print(colors[i])

                    points_3d[i] = Point3D(
                        id=i,
                        xyz=points[i],
                        rgb=colors[i] if has_color else [255, 255, 255]
                    )
        return points_3d
        

    def add_points(self, points3D, remove_statistical_outlier=True):
        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for idx, point3D in points3D.items():
            # if (idx > 10000):
            #     break
            # track_len = len(point3D.point2D_idxs)
            # if track_len < min_track_len:
            #     continue
            xyz.append(point3D.xyz)
            rgb.append(point3D.rgb / 255.)

        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
        # 获取渲染选项并设置点大小（默认是 5.0，可以设小一点）
        render_option = self.__vis.get_render_option()
        render_option.point_size = 1.0  # 越小越精细，默认是5.0

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize COLMAP binary and text models"
    )
    parser.add_argument(
        "--input_hdmap", required=True, help="path to input hdmap.ply"
    )
    parser.add_argument(
        "--input_ply", required=True, help="path to input *.ply"
    )

    args = parser.parse_args()
    return args

def example_cal(input_path):
        # 示例数据 - 替换为您的实际匹配点对
    # 来自3DGS重建点云的匹配点(N×3数组)
    source_points_1 = np.array([
        [-143.367, 257.409, 19.504],
        [51.026, -219.746, 33.969],
        [24.164, -27.88, 25.42]
    ])
    source_points_3 = np.array([
        [-161.329, -49.516, 11.249],
        [143.710, 37.432, 12.156],
        [-31.596, 36.47, 12.781],
    ])
    source_points_4 = np.array([
        [60.813, 87.547, 13.699],
        [92.623, 93.699, 13.529],
        [94.629, 63.52, 13.674],
        [61.998, 57.441, 13.832],
        [-92.086, -238.005, 13.495],
    ])
    source_points_5 = np.array([
        [-0.947016, -1.473178, 0.128795],
        [-0.705810, -1.492827, 0.127032],
        [-0.667408, -1.791514, 0.123960],
        [-0.863338, 0.620276, 0.136093],
    ])
    source_points_6 = np.array([
        # 右下方路口
        [225.962, -11.286, -225.056], # 右
        [213.202, -10.358, -253.802], # 下
        [189.700, -10.293, -239.180], # 左
        [202.384, -11.326, -210.185], # 上

        # 上方路口
        [-261.002, -13.345, 233.919], # 右
        [-259.480, -12.855, 204.191], # 下
        [-286.458, -12.025, 196.748], # 左
    ])
    
    source_points_7 = np.array([
        # 左下角路口
        [91.684, -25.149, 145.936], # 右
        [93.753, -25.1176, 173.023],# 下
        [122.449, -24.344, 168.033],# 左
        [116.393, -24.552, 138.337],# 上

        # 右下角路口
        [-171.176, -30.982, 215.917],
        [-155.277, -30.541, 238.130],
        [-137.792, -30.364, 221.542],
        [-153.684, -30.773, 199.032],

        # 上方路口
        [162.421, -24.423, -261.997],
        [166.512, -24.456, -236.357],
        [190.899, -23.903, -234.808],
    ])

    # reg1_40
    # source_points_8 = np.array([
    #     # 左下角路口
    #     [-102.628, -17.675, 263.452], # 右
    #     [-126.497, -17.512, 293.271],# 下
    #     [-152.069, -19.526, 274.442],# 左
    #     [-126.181, -19.538, 245.222],# 上

    #     # 上方路口
    #     [91.747, -31.384, -208.650],
    #     [69.634, -31.434, -185.981],
    #     [43.922, -33.604, -201.956],
    # ])
    source_points_8 = np.array([
        # 左下角路口
        [-143.86,-19.99,252.73],[-143.86,-18.78,285.74],[-109.95,-16.98,285.74],[-109.95,-19.39,249.43],

        # 上方路口
        [50.33,-34.45,-222.67],[53.42,-32.65, -192.96],[81.16,-30.84,-192.96]
    ])

    source_points_9 = np.array([
        # 左下角路口
        [-174.953, -11.839, -19.381], # 右
        [-160.393, -10.922, -49.341],# 下
        [-193.791, -10.626, -61.791],# 左
        [-206.689, -11.741, -27.087],# 上

        # 右下角路口
        [147.336, -13.056, 68.300],
        [144.254, -12.046, 34.819],
        [115.416, -12.522, 41.663],
        [118.381, -13.642, 75.341],])

    # 创建目标点(对源点应用一个已知变换)
    # true_scale = 1.5
    # true_R = np.array([
    #     [0.0, -1.0, 0.0],
    #     [1.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0]
    # ])
    # true_t = np.array([1.0, 2.0, 3.0])
    # target_points = true_scale * (source_points @ true_R.T) + true_t
    
    # 应用真实变换创建目标点,来自高精地图点云的对应点(N×3数组)

    target_points_1 = np.array([
        [211.725, -198.188, 0.],
        [510.802, 175.852, 0.],
        [368.399, 66.185, 0.],
    ])
    target_points_3 = np.array([
        # cross 1
        [184.440, -214.749, 0.],
        # cross 2
        [321.911, -486.929, 0.],
        # cross on road
        [294.709, -325.727, 0.],
    ])
    target_points_4 = np.array([
        # cross 2
        [353.667, -483.508, 0.],
        [321.911, -486.929, 0.],
        [322.876, -458.594, 0.],
        [355.031, -454.903, 0.],

        [544.419, -190.258, 0.],
    ])
    target_points_5 = np.array([
        # cross 3
        [510.802, 175.852, 0.],
        [485.786, 165.564, 0.],
        [471.584, 187.430, 0.],

        # cross 4
        # [368.399, 66.185, 0.],

        # cross on road
        # [294.709, -325.727, 0.],
        # [275.990, -334.921, 0.],

        # [544.419, -190.258, 0.],
        [613.978, -10.582, 0.]

    ])

    # seg 4+5
    target_points_6 = np.array([
        # cross 2
        [353.667, -483.508, 0.],
        [321.911, -486.929, 0.],
        [322.876, -458.594, 0.],
        [355.031, -454.903, 0.],

        # cross 3
        [510.802, 175.852, 0.],
        [485.786, 165.564, 0.],
        [471.584, 187.430, 0.],
    ])

    # seg 1+2+3
    target_points_7 = np.array([
        # cross 1
        [211.725, -198.188, 0.],
        [184.440, -214.749, 0.],
        [170.678, -186.497, 0.],
        [198.697, -169.936, 0.],

        # cross 2
        [353.667, -483.508, 0.],
        [321.911, -486.929, 0.],
        [322.876, -458.594, 0.],
        [355.031, -454.903, 0.],

        # cross 3
        [510.802, 175.852, 0.],
        [485.786, 165.564, 0.],
        [471.584, 187.430, 0.],
    ])

    # 0712_cam3_DJI_0309_reg1_40_seq_pcl.ply 
    target_points_8 = np.array([
        # cross 1
        [211.725, -198.188, 0.],
        [184.440, -214.749, 0.],
        [170.678, -186.497, 0.],
        [198.697, -169.936, 0.],

        # cross 3
        [510.802, 175.852, 0.],
        [485.786, 165.564, 0.],
        [471.584, 187.430, 0.],
    ])

    # 
    target_points_9 = np.array([
        # cross 1
        [211.725, -198.188, 0.],
        [184.440, -214.749, 0.],
        [170.678, -186.497, 0.],
        [198.697, -169.936, 0.],

        # cross 2
        [353.667, -483.508, 0.],
        [321.911, -486.929, 0.],
        [322.876, -458.594, 0.],
        [355.031, -454.903, 0.],
    ])
    
    # 计算变换
    s, R, t, transformation_matrix = compute_similarity_transformation(source_points_8, target_points_8)
    
    print("计算得到的缩放因子:", s)
    print("\n计算得到的旋转矩阵:")
    print(R)
    print("\n计算得到的平移向量:")
    print(t)
    print("\n4x4齐次变换矩阵:")
    print(transformation_matrix)
    
    # 保存变换矩阵到文件
    save_transformation_matrix("transformation_matrix.txt", transformation_matrix, 
                             "从3DGS到高精地图的变换矩阵")
    save_transformation_json("transform_matrix.json", transformation_matrix, input_path)
    
    return transformation_matrix
    # # 验证变换
    # transformed_source = apply_transformation(source_points, transformation_matrix)
    # error = np.mean(np.linalg.norm(transformed_source - target_points, axis=1))
    # print(f"\n配准误差(平均点距离): {error:.6f}")
    
    # # 输出变换后的点云(示例)
    # print("\n变换后的前5个点(示例):")
    # print(transformed_source[:3])

def main():
    args = parse_args()

    # read COLMAP model
    visualizer = Visualizer()
    hdmap = visualizer.read_ply(args.input_hdmap)

    if (".json" in args.input_ply) : # 输入json，制作所有点云的拼接结果
        idx = len(hdmap)
        with open(args.input_ply) as ply_info_file:
            ply_infos = json.load(ply_info_file)
            for seg, info in ply_infos.items():
                ply = visualizer.read_ply(info.get("source_ply", ""))
                trans_matrix = info.get("transformation_matrix", None)
                trans_matrix = np.array(trans_matrix)
                for _, point3d in ply.items():
                    transformed_ply = apply_transformation(np.array([point3d.xyz]), transformation_matrix=trans_matrix)

                    hdmap[idx] = Point3D(
                        id=idx,
                        xyz=transformed_ply[0],
                        rgb=point3d.rgb
                    )
                    idx += 1
    else:
        ply = visualizer.read_ply_new(args.input_ply) # 输入单个文件，计算求解关系

        print("num_hdmap:", len(hdmap))
        print("num_points:", len(ply))

        trans_matrix = example_cal(args.input_ply)

        start_idx = len(hdmap)
        for idx, point3d in ply.items():
            transformed_ply = apply_transformation(np.array([point3d.xyz]), transformation_matrix=trans_matrix)

            hdmap[start_idx + idx] = Point3D(
                id=start_idx+idx,
                xyz=transformed_ply[0],
                rgb=point3d.rgb
            )

    # display using Open3D visualization tools
    visualizer.create_window()
    visualizer.add_points(hdmap)
    visualizer.show()

# 示例使用
if __name__ == "__main__":
    main()