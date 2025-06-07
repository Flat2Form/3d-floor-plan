import os
import open3d as o3d
import numpy as np
import random
from sklearn.cluster import KMeans
import argparse
import matplotlib.pyplot as plt
def find_corner_points(points, plane_model):
    """평면의 방향을 고려하여 꼭짓점을 찾는 함수"""
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    
    # 평면의 기저 벡터 계산
    # 첫 번째 기저 벡터는 z축과의 외적
    basis1 = np.cross(normal, [0, 0, 1])
    if np.all(basis1 == 0):  # 평면이 수평인 경우(평면의 법선벡터가 z축과 평행할 때)
        basis1 = np.cross(normal, [0, 1, 0])
    basis1 = basis1 / np.linalg.norm(basis1)
    
    # 두 번째 기저 벡터
    basis2 = np.cross(normal, basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    
    # 점들을 새로운 2D 좌표계로 투영
    points_2d = np.zeros((len(points), 2))
    for i, point in enumerate(points):
        points_2d[i, 0] = np.dot(point, basis1)
        points_2d[i, 1] = np.dot(point, basis2)
    
    # 2D 투영된 점들의 경계 찾기
    min_x, min_y = np.min(points_2d, axis=0)
    max_x, max_y = np.max(points_2d, axis=0)
    
    # 2D 좌표계에서의 꼭짓점
    corners_2d = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
    
    # 다시 3D로 변환
    corners_3d = np.zeros((4, 3))
    for i, corner in enumerate(corners_2d):
        # 기저 벡터를 이용해 3D 좌표 계산
        p = corner[0] * basis1 + corner[1] * basis2
        # 평면 위의 점으로 투영
        t = -(np.dot(p, normal) + d) / np.dot(normal, normal)
        corners_3d[i] = p + t * normal
    
    return corners_3d

def create_plane_points(corners_3d, num_points=50):
    """코너 점들 사이를 보간하여 평면 포인트 생성"""
    points_per_side = int(np.sqrt(num_points))
    
    # 평면의 포인트 생성
    grid_points = []
    for i in range(points_per_side):
        # top과 bottom 엣지 사이를 보간
        t = i / (points_per_side - 1)
        left = corners_3d[0] * (1-t) + corners_3d[3] * t
        right = corners_3d[1] * (1-t) + corners_3d[2] * t
        
        for j in range(points_per_side):
            # left와 right 포인트 사이를 보간
            s = j / (points_per_side - 1)
            point = left * (1-s) + right * s
            grid_points.append(point)
    
    return np.array(grid_points)

def analyze_distribution(points):
    """점들의 분포를 분석하여 클러스터링이 필요한지 판단"""
    # 중심점으로부터의 거리 계산
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    
    # 거리의 표준편차 계산
    std_dev = np.std(distances)
    mean_dist = np.mean(distances)
    
    # 변동계수(CV) 계산 - 데이터의 상대적인 분산 정도를 나타냄
    cv = std_dev / mean_dist if mean_dist > 0 else 0
    
    # CV가 임계값을 넘으면 클러스터링이 필요하다고 판단
    return cv > 0.5  # 임계값은 실험적으로 조정 가능

def filter_largest_cluster(pcd, eps=0.1, min_points=10, min_valid_ratio=0.04):
    """DBSCAN을 사용하여 너무 작은 클러스터는 노이즈로 간주하고, 남은 클러스터는 모두 합쳐 반환"""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if len(labels) == 0:
        return None, pcd
    print(f"전체 포인트 개수: {len(labels)}, 클러스터 개수: {len(np.unique(labels[labels >= 0]))}")
    # 가장 많은 포인트를 가진 클러스터 찾기
    # labels >= 0: 노이즈 포인트(-1)를 제외한 실제 클러스터만 선택
    # unique_labels: 클러스터 레이블 목록
    # counts: 각 클러스터에 속한 포인트 개수
    total_points = len(labels)

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    
    # 너무 작은 클러스터(전체의 5% 미만)는 노이즈로 간주
    min_valid_points = int(total_points * min_valid_ratio)
    valid_labels = unique_labels[counts >= min_valid_points]
    # --- 클러스터가 2개 이상이면 각 클러스터를 시각화 ---
    # if len(unique_labels) > 1:
    #     geometries = []
    #     colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(unique_labels)))[:, :3]
    #     for idx, label in enumerate(unique_labels):
    #         indices = np.where(labels == label)[0]
    #         cluster = pcd.select_by_index(indices)
    #         cluster.paint_uniform_color(colors[idx % len(colors)])
    #         geometries.append(cluster)
    #     # 노이즈(-1)도 시각화
    #     noise_indices = np.where(labels == -1)[0]
    #     if len(noise_indices) > 0:
    #         noise = pcd.select_by_index(noise_indices)
    #         noise.paint_uniform_color([0.5, 0.5, 0.5])
    #         geometries.append(noise)
    #     o3d.visualization.draw_geometries(geometries)
    # ---------------------------------------------------

    # 유효한 클러스터가 없으면 원본 포인트 클라우드 반환
    if len(counts) == 0 or len(valid_labels) == 0:
        return None, pcd
    
    # 남은 클러스터가 2개 이상이면 모두 합쳐서 반환
    valid_indices = []
    for label in valid_labels:
        indices = np.where(labels == label)[0]
        valid_indices.extend(indices)
    rest_indices = [i for i in range(len(labels)) if labels[i] not in valid_labels]
    if len(valid_indices) > 0:
        return pcd.select_by_index(valid_indices), pcd.select_by_index(rest_indices)
    else:
        return None, pcd

def filter_by_distance_to_center(pcd, max_distance):
    """중심으로부터 일정 거리 이상 떨어진 점들 제거"""
    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    valid_indices = np.where(distances < max_distance)[0]
    return pcd.select_by_index(valid_indices)

def remove_noise(pcd, nb_neighbors=20, std_ratio=1.5):
    """여러 단계의 노이즈 제거 적용"""
    # 1. Statistical outlier removal
    clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    # 2. DBSCAN으로 가장 큰 클러스터 선택 (min_points를 None으로 넘겨서 30% 적용)
    valid_pcd, rest_pcd = filter_largest_cluster(clean_pcd, eps=0.1, min_points=10)
    return valid_pcd, rest_pcd

def detect_plane(pcd, distance_threshold=0.02, min_inliers=3000):
    """하나의 평면을 검출하는 함수"""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    
    if len(inliers) < min_inliers:
        return None, None, pcd, False
    
    inlier_cloud = pcd.select_by_index(inliers)
    # Statistical outlier removal 적용
    inlier_cloud, noise = remove_noise(inlier_cloud)
    outliers = pcd.select_by_index(inliers, invert=True)
    if noise is not None:
        rest = outliers + noise
    else:
        rest = outliers
    return plane_model, inlier_cloud, rest, True

def create_plane_mesh(inlier_cloud, plane_model, num_points=50000):
    """평면의 메쉬(포인트 클라우드)를 생성하는 함수"""
    inlier_points = np.asarray(inlier_cloud.points)
    corners_3d = find_corner_points(inlier_points, plane_model)
    
    new_plane_points = create_plane_points(corners_3d, num_points)
    new_plane_pcd = o3d.geometry.PointCloud()
    new_plane_pcd.points = o3d.utility.Vector3dVector(new_plane_points)
    new_plane_pcd.paint_uniform_color([random.random(), random.random(), random.random()])
    
    return new_plane_pcd, corners_3d

def make_wall(pcd, max_planes=30):

    pcd = pcd.voxel_down_sample(voxel_size=0.005)

    # 여러 평면 검출 파라미터
    max_planes = max_planes
    min_inliers = len(pcd.points) * 0.03
    distance_threshold = 0.02
    rest = pcd
    plane_meshes = []
    outlier_cloud = None
    
    # 평면 검출 반복
    for i in range(max_planes):
        print(f"\n{i+1}번째 평면을 검출하는 중...")
        
        # 평면 검출
        plane_model, inlier_cloud, rest, success = detect_plane(
            rest, 
            distance_threshold=distance_threshold,
            min_inliers=min_inliers
        )

        if not success:
            print(f"{i+1}번째 평면: 검출 실패")
            outlier_cloud = rest
            break
        # 평면 메쉬 생성
        plane_mesh, corners = create_plane_mesh(inlier_cloud, plane_model)
        plane_meshes.append(plane_mesh)

        outlier_cloud = rest
    
    geometries = plane_meshes
    return geometries, outlier_cloud