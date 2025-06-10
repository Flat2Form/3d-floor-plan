import open3d as o3d
import numpy as np
import os
import argparse

def save_pcd(pcd, path):
    o3d.io.write_point_cloud(path, pcd)

def load_pcd(path):
    return o3d.io.read_point_cloud(path)

def save_ply(pcd, path):
    o3d.io.write_point_cloud(path, pcd)

def load_ply(path):
    return o3d.io.read_point_cloud(path)

def pcd_list_to_pcd(pcd_list):
    combined = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        combined += pcd
    return combined

def obb_to_pcd(obb):
    """
    OBB의 모든 엣지 위를 주어진 간격(spacing)으로 샘플링한 PointCloud를 생성합니다.
    
    Args:
        obb (o3d.geometry.OrientedBoundingBox): 샘플링할 OrientedBoundingBox
        spacing (float): 엣지 위 두 점 사이의 샘플링 간격 (단위: 동일 좌표계 단위)
                       함수 내부에서 기본값 0.01로 설정
    
    Returns:
        open3d.geometry.PointCloud: 생성된 포인트클라우드
    """
    # sampling 간격 (원하는 값으로 바꿀 수 있습니다)
    spacing = 0.01

    # OBB로부터 엣지 LineSet 생성
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    verts = np.asarray(line_set.points)
    edges = np.asarray(line_set.lines)

    # 각 엣지를 spacing 간격으로 샘플링
    sampled_pts = []
    for (i, j) in edges:
        p1, p2 = verts[i], verts[j]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        # 최소 하나의 segment는 생성
        num_seg = max(int(np.floor(edge_len / spacing)), 1)
        ts = np.linspace(0, 1, num_seg + 1)
        for t in ts:
            sampled_pts.append(p1 + t * edge_vec)

    sampled_pts = np.vstack(sampled_pts)
    # PointCloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_pts)
    return pcd

def create_aabb_list(pcd, instance_labels):
    """포인트 클라우드와 레이블을 입력받아 바운딩 박스 리스트를 생성하는 함수
    Args:
        pcd: 포인트 클라우드 객체 (open3d.geometry.PointCloud)
        labels: 각 포인트의 레이블 (길이 N의 numpy 배열)
    Returns:
        obb_list: 바운딩 박스 객체들의 리스트 (list[open3d.geometry.OrientedBoundingBox])
    """
    coords = np.asarray(pcd.points)
    aabb_list = []
    
    # 각 고유한 레이블에 대해 처리
    for label in np.unique(instance_labels):
        if label < 0:  # 유효하지 않은 레이블 무시
            continue
            
        # 현재 레이블에 해당하는 포인트들 추출
        mask = instance_labels == label
        label_coords = coords[mask]
        
        # 포인트가 없는 경우 건너뛰기
        if len(label_coords) == 0:
            continue
            
        # 포인트 클라우드 생성
        label_pcd = o3d.geometry.PointCloud()
        label_pcd.points = o3d.utility.Vector3dVector(label_coords)
        
        # 바운딩 박스 생성 및 색상 설정
        aabb = label_pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)  # 기본 색상 설정
        
        aabb_list.append(aabb)
    
    return aabb_list

def create_obb_list(pcd, instance_labels):
    """포인트 클라우드와 레이블을 입력받아 바운딩 박스 리스트를 생성하는 함수
    Args:
        pcd: 포인트 클라우드 객체 (open3d.geometry.PointCloud)
        labels: 각 포인트의 레이블 (길이 N의 numpy 배열)
    Returns:
        obb_list: 바운딩 박스 객체들의 리스트 (list[open3d.geometry.OrientedBoundingBox])
    """
    coords = np.asarray(pcd.points)
    obb_list = []
    
    # 각 고유한 레이블에 대해 처리
    for label in np.unique(instance_labels):
        if label < 0:  # 유효하지 않은 레이블 무시
            continue
            
        # 현재 레이블에 해당하는 포인트들 추출
        mask = instance_labels == label
        label_coords = coords[mask]
        
        # 포인트가 없는 경우 건너뛰기
        if len(label_coords) == 0:
            continue
            
        # 포인트 클라우드 생성
        label_pcd = o3d.geometry.PointCloud()
        label_pcd.points = o3d.utility.Vector3dVector(label_coords)
        
        # 바운딩 박스 생성 및 색상 설정
        obb = label_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  # 기본 색상 설정
        
        obb_list.append(obb)
    
    return obb_list

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
    for i in range(len(np.unique(labels))):
        if i == -1:  # 노이즈 클러스터는 건너뛰기
            continue
        cluster_pcd = pcd.select_by_index(np.where(labels == i)[0])
        if len(cluster_pcd.points) < 100:
            print(f"cluster {i}({len(cluster_pcd.points)}) is too small")
    return labels

def create_3d_floor_plan(wall_planes_vertices, obb_list):
    """
    벽면 폴리곤과 OBB 리스트를 입력받아 3D 평면도를 생성합니다.

    Args:
        wall_planes_vertices (List[np.ndarray]): 각 벽면의 꼭짓점들 (N_i x 3 배열)의 리스트
        obb_list (List[o3d.geometry.OrientedBoundingBox]): 객체의 OBB 리스트

    Returns:
        List[o3d.geometry.Geometry3D]: 벽면 메쉬와 OBB 포인트클라우드가 포함된 지오메트리 리스트
    """
    pcds = []

    # 1. 벽면 포인트클라우드 생성
    for verts in wall_planes_vertices:
        pcd = quad_to_pcd(verts)
        pcd.paint_uniform_color((0.8, 0.8, 0.8))
        pcds.append(pcd)

    # 2. OBB마다 포인트클라우드로 변환
    for obb in obb_list:
        pcd = obb_to_pcd(obb)
        pcd.paint_uniform_color((1, 0, 0))
        pcds.append(pcd)

    return pcds

def quad_to_pcd(verts):
    """
    4개 꼭짓점으로 정의된 사각형 내부를 주어진 간격(du, dv)으로 샘플링한 PointCloud를 생성합니다.
    
    Args:
        verts (array-like): shape (4,3)인 3D 꼭짓점 배열. 
            순서: v0→v1→v2→v3이 사각형을 한 바퀴 돌아야 합니다.
        du (float): v0→v1 방향(첫 번째 변)으로의 샘플링 간격 (단위: 같은 좌표계 단위)
        dv (float): v0→v3 방향(네 번째 변)으로의 샘플링 간격
    
    Returns:
        open3d.geometry.PointCloud: 생성된 포인트클라우드
    """
    du = 0.03
    dv = 0.03
    v0, v1, v2, v3 = [np.array(v, dtype=float) for v in verts]
    # 두 방향 벡터와 길이
    eu = v1 - v0
    ev = v3 - v0
    Lu = np.linalg.norm(eu)
    Lv = np.linalg.norm(ev)
    # 실제 파라미터 간격
    ds = du / Lu
    dt = dv / Lv
    ss = np.arange(0, 1 + 1e-6, ds)
    ts = np.arange(0, 1 + 1e-6, dt)
    
    pts = []
    for s in ss:
        for t in ts:
            # bilinear 보간
            p = (1-s)*(1-t)*v0 + s*(1-t)*v1 + s*t*v2 + (1-s)*t*v3
            pts.append(p)
    pts = np.vstack(pts)
    
    # PointCloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd