# Open3D의 RegionGrowing (ver.0.15+)
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from make_wall import make_wall

def object_detection(pcd):
    """
    입력: pcd (open3d.geometry.PointCloud)
    출력: labels (np.ndarray) - 각 포인트의 클러스터 레이블
    min_cluster_size: 최소 클러스터 크기(포인트 수)
    """
    # 노멀 추정 (DBSCAN에는 필수는 아니지만, 후처리에 유용)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    min_cluster_size = len(pcd.points) * 0.01
    # DBSCAN 클러스터링
    labels = np.array(
        pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=True)
    )
    # 작은 클러스터 제거
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        if label == -1:
            continue
        if count < min_cluster_size:
            labels[labels == label] = -1
    return labels

if __name__ == "__main__":
    # 1. 포인트 클라우드 로드
    pcd = o3d.io.read_point_cloud("scaniverse.ply")
    geometries, pcd = make_wall(pcd)
    print("PointCloud loaded:", pcd)

    # 2. 오브젝트 디텍션(클러스터링)
    labels = object_detection(pcd)

    mask = np.logical_and(labels != 0, labels != -1)
    indices = np.where(mask)[0]
    filtered_pcd = pcd.select_by_index(indices)

    o3d.visualization.draw_geometries([filtered_pcd])
    # labels를 npy 파일로 저장
    np.save('labels.npy', labels)
    # pcd를 ply 파일로 저장
    o3d.io.write_point_cloud('filtered_pcd.ply', pcd)
    print('레이블이 labels.npy 파일로 저장되었습니다.')