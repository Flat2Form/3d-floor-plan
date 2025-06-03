import argparse
import os
import os.path as osp
from operator import itemgetter
import open3d as o3d

import numpy as np

# yapf:disable
COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255
# yapf:enable

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array([
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture'
])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {
    1: 'wall',
    2: 'floor',
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    8: 'door',
    9: 'window',
    10: 'bookshelf',
    11: 'picture',
    12: 'counter',
    14: 'desk',
    16: 'curtain',
    24: 'refridgerator',
    28: 'shower curtain',
    33: 'toilet',
    34: 'sink',
    36: 'bathtub',
    39: 'otherfurniture'
}


def get_coords_color_obb(opt):

    coord_file = osp.join(opt.prediction_path, 'coords', opt.room_name + '.npy')
    color_file = osp.join(opt.prediction_path, 'colors', opt.room_name + '.npy')
    xyz = np.load(coord_file)
    rgb = np.load(color_file)
    rgb = (rgb + 1) * 127.5

    instance_file = os.path.join(opt.prediction_path, 'pred_instance', opt.room_name + '.txt')
    assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
    f = open(instance_file, 'r')
    masks = f.readlines()
    masks = [mask.rstrip().split() for mask in masks]
    # RGB 값을 저장할 배열 초기화
    inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #

    # 인스턴스 개수와 포인트 수 배열 초기화
    ins_num = len(masks)
    ins_pointnum = np.zeros(ins_num)
    inst_label = -100 * np.ones(rgb.shape[0]).astype(int)

    # 점수 기준으로 정렬하여 높은 점수가 시각화 우선순위를 가지도록 함
    scores = np.array([float(x[-1]) for x in masks])
    sort_inds = np.argsort(scores)[::-1]
    # 각 마스크에 대해 처리
    obb_list = []
    pcd_list = []
    for i_ in range(len(masks) - 1, -1, -1):
        i = sort_inds[i_]
        # 마스크 파일 경로 생성
        mask_path = os.path.join(opt.prediction_path, 'pred_instance', masks[i][0])
        assert os.path.isfile(mask_path), mask_path
        # 낮은 점수의 마스크는 건너뛰기
        if (float(masks[i][2]) < 0.09):
            continue
        # 마스크 파일 읽어서 처리
        mask = np.array(open(mask_path).read().splitlines(), dtype=int)
        # 마스크가 1인 포인트들의 인덱스 찾기
        point_indices = np.where(mask == 1)[0]
        # 마스크가 1인 포인트들의 xyz 좌표 추출
        mask_xyz = xyz[point_indices]  # shape: (N, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mask_xyz)

        # obb = pcd.get_axis_aligned_bounding_box()
        obb = pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)  # 초록색으로 설정
        
        # 클래스 ID를 클래스 이름으로 변환
        class_id = int(masks[i][1])
        class_name = SEMANTIC_IDX2NAME.get(class_id, "알 수 없음")
        
        # OBB와 PointCloud를 각각의 리스트에 추가
        obb_list.append(obb)
        pcd_list.append(pcd)
        
        print('{} {}: pointnum: {}'.format(i, masks[i], mask.sum()))
        ins_pointnum[i] = mask.sum()
        inst_label[mask == 1] = i
    # 포인트 수에 따라 정렬
    sort_idx = np.argsort(ins_pointnum)[::-1]
    # 각 인스턴스에 색상 할당
    for _sort_id in range(ins_num):
        inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[
            _sort_id % len(COLOR_DETECTRON2)]
    rgb = inst_label_pred_rgb

    # o3d.visualization.draw_geometries(pcd_list + obb_list)
    return xyz, rgb, obb_list


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2],
                                                            int(color[0] * 255),
                                                            int(color[1] * 255),
                                                            int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction_path', help='path to the prediction results', default='./results')
    parser.add_argument('--room_name', help='room_name', default='scene0011_00')
    parser.add_argument(
        '--task',
        help='input/semantic_gt/semantic_pred/offset_semantic_pred/instance_gt/instance_pred',
        default='instance_pred')
    parser.add_argument('--out', help='output point cloud file in FILE.ply format')
    opt = parser.parse_args()

    xyz, rgb, obb_list= get_coords_color_obb(opt)
    points = xyz[:, :3]
    colors = rgb / 255
    # to_glb(points, colors, obb_list)

    if opt.out != '':
        assert '.ply' in opt.out, 'output cloud file should be in FILE.ply format'
        write_ply(points, colors, None, opt.out)
    else:
        # Open3D를 사용하여 포인트 클라우드를 시각화하는 코드입니다.
        # 1. points와 colors 데이터로 포인트 클라우드 객체를 생성합니다.
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)  # 3D 좌표값 설정
        pc.colors = o3d.utility.Vector3dVector(colors)  # RGB 색상값 설정
        
        # 2. 시각화 윈도우를 생성하고 포인트 클라우드를 표시합니다.
        vis = o3d.visualization.Visualizer()
        vis.create_window()  # 시각화 윈도우 생성
        vis.add_geometry(pc)  # 포인트 클라우드 추가
        for obb in obb_list:
            vis.add_geometry(obb)  # 포인트 클라우드 추가
        vis.get_render_option().point_size = 1.5  # 포인트 크기 설정
        vis.run()  # 시각화 실행
        vis.destroy_window()  # 윈도우 종료
