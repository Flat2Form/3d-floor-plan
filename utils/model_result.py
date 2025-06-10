import os
import os.path as osp
import open3d as o3d
import numpy as np

def model_result_to_labels(room_name):
    """
    원본 get_coords_color_obb 함수를 수정하여, 각 포인트가 속한 인스턴스 ID를 담은
    1D numpy array (inst_labels)를 함께 반환합니다.

    Returns:
        xyz (np.ndarray): (N,3) 좌표
        rgb (np.ndarray): (N,3) 컬러
        obb_list (List[o3d.geometry.OrientedBoundingBox]): 감지된 객체 OBB 리스트
        wall_0, wall_1: 두 벽면용 PointCloud 리스트나 단일 PointCloud
        inst_labels (np.ndarray): (N,) 각 포인트의 인스턴스 번호, 
                                  인스턴스가 없으면 -1
    """
    semantic_file = osp.join("results", 'semantic_pred', room_name + '.npy')
    assert os.path.isfile(semantic_file), f'No semantic result - {semantic_file}.'
    label_pred = np.load(semantic_file).astype(int)  # 0~19
    
    coord_file = osp.join("results", 'coords', room_name + '.npy')
    assert os.path.isfile(coord_file), f'No coord file - {coord_file}.'
    xyz = np.load(coord_file)
    labels = -100 * np.ones(xyz.shape[0]).astype(int)

    color_file = osp.join("results", 'colors', room_name + '.npy')
    assert os.path.isfile(color_file), f'No color file - {color_file}.'
    rgb = (np.load(color_file) + 1) * 127.5

    instance_txt = osp.join("results", 'pred_instance', room_name + '.txt')
    assert os.path.isfile(instance_txt), f'No instance result - {instance_txt}.'
    f = open(instance_txt, 'r')
    masks = f.readlines()
    masks = [mask.rstrip().split() for mask in masks]

    # 점수 기준으로 정렬하여 높은 점수가 시각화 우선순위를 가지도록 함
    scores = np.array([float(x[-1]) for x in masks])
    sort_inds = np.argsort(scores)[::-1]
    inst_id = 103
    # 각 마스크에 대해 처리
    for i_ in range(len(masks) - 1, -1, -1):
        i = sort_inds[i_]
        # 마스크 파일 경로 생성
        mask_path = os.path.join("results", 'pred_instance', masks[i][0])
        assert os.path.isfile(mask_path), mask_path
        print(masks[i])
        # 낮은 점수의 마스크는 건너뛰기
        if (float(masks[i][2]) < 0.09):
            print("점수 낮음")
            continue
        # 마스크 파일 읽어서 처리
        mask = np.array(open(mask_path).read().splitlines(), dtype=int)
        # 마스크가 1인 포인트들의 인덱스 찾기
        point_indices = np.where(mask == 1)[0]
        if len(point_indices) < xyz.shape[0] * 0.02:
            print("포인트 수 적음")
            continue
        labels[point_indices] = inst_id
        inst_id += 1

        print('{} {}: pointnum: {}'.format(i, masks[i], mask.sum()))
    class_0_idx = np.where(label_pred == 0)[0]
    class_1_idx = np.where(label_pred == 1)[0]
    labels[class_0_idx] = -2
    labels[class_1_idx] = -3
    return xyz, rgb, labels

if __name__ == "__main__":
    room_name = "scaniverse-model"
    xyz, rgb, labels = get_coords_with_instance_labels(room_name)
    from utils import create_obb_list
    import open3d as o3d
    pcd = o3d.io.read_point_cloud("utils/scaniverse.ply")
    obb_list = create_obb_list(pcd, labels)
    o3d.visualization.draw_geometries([pcd] + obb_list)
    np.save("labels.npy", labels)
    print(labels)