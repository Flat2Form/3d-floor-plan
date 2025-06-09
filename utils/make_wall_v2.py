import open3d as o3d
import numpy as np

def segment_plane(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """
    RANSAC을 사용하여 포인트 클라우드에서 평면을 분할합니다.
    
    매개변수:
        pcd: open3d.geometry.PointCloud - 입력 포인트 클라우드
        distance_threshold: float - RANSAC 평면 검출을 위한 거리 임계값 (기본값: 0.02)
        ransac_n: int - RANSAC 샘플링 포인트 수 (기본값: 3)
        num_iterations: int - RANSAC 반복 횟수 (기본값: 1000)
    
    반환값:
        model: (a,b,c,d) - 평면 방정식 ax+by+cz+d=0의 계수
        normal: (3,) array - 정규화된 평면의 법선 벡터
        inliers: array - 평면에 속하는 점들의 인덱스
    """
    model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    a, b, c, d = model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    return model, normal, inliers

def filter_largest_cluster(pcd, eps=0.1, min_points=10, min_valid_ratio=0.04):
    """
    DBSCAN 클러스터링을 사용하여 가장 큰 클러스터를 필터링합니다.
    
    매개변수:
        pcd: open3d.geometry.PointCloud - 입력 포인트 클라우드
        eps: float - DBSCAN 클러스터링의 이웃 탐색 반경 (기본값: 0.1)
        min_points: int - DBSCAN 클러스터의 최소 점 개수 (기본값: 10)
        min_valid_ratio: float - 유효 클러스터의 최소 크기 비율 (기본값: 0.04)
    
    반환값:
        (valid_cloud, rest_cloud): (PointCloud, PointCloud) - 유효한 클러스터와 나머지 점들
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.size == 0:
        return pcd, pcd
    total = labels.size
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    min_pts = int(total * min_valid_ratio)
    valid = unique[counts >= min_pts]
    if valid.size == 0:
        return pcd, pcd
    idx = np.hstack([np.where(labels == lbl)[0] for lbl in valid])
    rest = np.setdiff1d(np.arange(total), idx)
    return pcd.select_by_index(idx), pcd.select_by_index(rest)

def remove_noise(pcd):
    """
    통계적 이상치 제거와 DBSCAN을 사용하여 노이즈를 제거합니다.
    
    매개변수:
        pcd: open3d.geometry.PointCloud - 입력 포인트 클라우드
    
    반환값:
        (valid_cloud, rest_cloud): (PointCloud, PointCloud) - 노이즈가 제거된 클라우드와 노이즈 점들
    """
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    valid, rest = filter_largest_cluster(clean)
    return valid, rest

def detect_plane(pcd, distance_threshold=0.02, min_inliers=3000):
    """
    최소 인라이어 조건을 만족하는 하나의 평면을 검출합니다.
    
    매개변수:
        pcd: open3d.geometry.PointCloud - 입력 포인트 클라우드
        distance_threshold: float - RANSAC 평면 검출을 위한 거리 임계값 (기본값: 0.02)
        min_inliers: int - 최소 인라이어 개수 (기본값: 3000)
    
    반환값:
        model: (a,b,c,d) or None - 검출된 평면의 방정식 계수
        normal: (3,) array or None - 평면의 법선 벡터
        inlier_cloud: PointCloud or None - 평면에 속하는 점들의 클라우드
        remainder: PointCloud - 남은 점들의 클라우드
        ok: bool - 평면 검출 성공 여부
    """
    model, normal, inliers = segment_plane(pcd, distance_threshold)
    if len(inliers) < min_inliers:
        return None, None, None, pcd, False
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud, noise = remove_noise(inlier_cloud)
    remainder = pcd.select_by_index(inliers, invert=True)
    if noise is not None and len(noise.points) > 0:
        remainder = remainder + noise
    return model, normal, inlier_cloud, remainder, True

def find_corner_points(points, model):
    """
    평면 상의 점들로부터 경계 사각형의 꼭지점들을 찾습니다.
    
    매개변수:
        points: (N,3) array - 3D 점들의 좌표
        model: (a,b,c,d) - 평면 방정식의 계수
    
    반환값:
        corners: (4,3) array - 평면 상의 경계 사각형 꼭지점들의 3D 좌표
    """
    a, b, c, d = model
    normal = np.array([a, b, c])
    basis1 = np.cross(normal, [0, 0, 1])
    if np.allclose(basis1, 0):
        basis1 = np.cross(normal, [0, 1, 0])
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)
    basis2 /= np.linalg.norm(basis2)
    pts2d = np.column_stack((points.dot(basis1), points.dot(basis2)))
    mn, mx = pts2d.min(axis=0), pts2d.max(axis=0)
    corners2d = np.array([
        [mn[0], mn[1]],
        [mx[0], mn[1]],
        [mx[0], mx[1]],
        [mn[0], mx[1]]
    ])
    corners3d = []
    for x, y in corners2d:
        p = basis1 * x + basis2 * y
        t = -(p.dot(normal) + d) / normal.dot(normal)
        corners3d.append(p + t * normal)
    return np.array(corners3d)

def compute_missing_corners(idx, plane_models, plane_normals, plane_clouds, tol_rad, ext_ratio=0.0):
    """
    두 평면의 교선을 계산하고 연장하여 네 개의 꼭지점을 생성합니다.
    
    매개변수:
        idx: int - 기준 평면의 인덱스
        plane_models: list - 평면 방정식 계수들의 리스트
        plane_normals: list - 평면 법선 벡터들의 리스트
        plane_clouds: list - 평면 포인트 클라우드들의 리스트
        tol_rad: float - 수직 판정을 위한 각도 허용 오차(라디안)
        ext_ratio: float - 교선 연장 비율 (기본값: 0.0)
    
    반환값:
        corners: (4,3) array or None - 계산된 네 꼭지점의 좌표
    """
    num = len(plane_models)
    cand = [j for j in range(num)
            if j != idx and abs(np.dot(plane_normals[idx], plane_normals[j])) < np.cos(tol_rad)]
    if len(cand) < 2:
        return None
    j, k = cand[0], cand[1]

    def line_points(n_i, n_j, d_i, d_j, pts_cloud):
        A2 = np.vstack((n_i, n_j))
        d2 = np.array([d_i, d_j])
        p_base, _, _, _ = np.linalg.lstsq(A2, -d2, rcond=None)
        dir_line = np.cross(n_i, n_j)
        dir_line /= np.linalg.norm(dir_line)
        t_vals = (pts_cloud - p_base).dot(dir_line)
        t_min, t_max = t_vals.min(), t_vals.max()
        ext = (t_max - t_min) * ext_ratio
        return (p_base + (t_min - ext) * dir_line,
                p_base + (t_max + ext) * dir_line)

    pts_cloud = np.asarray(plane_clouds[idx].points)
    p1, p2 = line_points(
        plane_normals[idx], plane_normals[j],
        plane_models[idx][3], plane_models[j][3], pts_cloud
    )
    p3, p4 = line_points(
        plane_normals[idx], plane_normals[k],
        plane_models[idx][3], plane_models[k][3], pts_cloud
    )
    return np.array([p1, p2, p3, p4])

from itertools import combinations

def find_plane_corners_by_planes(plane_models, plane_normals, plane_clouds, angle_tol=10.0):
    """
    여러 평면의 교차점을 계산하여 각 평면의 모서리 점들을 찾습니다.
    
    매개변수:
        plane_models: list of (a,b,c,d) - 평면들의 방정식 계수
        plane_normals: list of (3,) array - 평면들의 법선 벡터
        plane_clouds: list of PointCloud - 평면들의 포인트 클라우드
        angle_tol: float - 수직 판정을 위한 각도 허용 오차 (기본값: 10.0)
    
    반환값:
        corners: (N,4,3) array - N개 평면의 각각 4개 모서리 점 좌표
    """
    tol_rad = np.deg2rad(angle_tol)
    num = len(plane_models)
    corners_per_plane = [[] for _ in range(num)]
    for (i, j, k) in combinations(range(num), 3):
        n1, n2, n3 = plane_normals[i], plane_normals[j], plane_normals[k]
        ang12 = np.arccos(np.clip(abs(np.dot(n1, n2)), -1.0, 1.0))
        ang13 = np.arccos(np.clip(abs(np.dot(n1, n3)), -1.0, 1.0))
        ang23 = np.arccos(np.clip(abs(np.dot(n2, n3)), -1.0, 1.0))
        if abs(ang12 - np.pi/2) < tol_rad and abs(ang13 - np.pi/2) < tol_rad and abs(ang23 - np.pi/2) < tol_rad:
            A = np.vstack((plane_models[i][:3], plane_models[j][:3], plane_models[k][:3]))
            d = np.array([plane_models[i][3], plane_models[j][3], plane_models[k][3]])
            try:
                pt = np.linalg.solve(A, -d)
            except np.linalg.LinAlgError:
                continue
            for idx in (i, j, k):
                corners_per_plane[idx].append(pt)
    for idx in range(num):
        if len(corners_per_plane[idx]) < 4:
            print(len(corners_per_plane[idx]))
            fallback = compute_missing_corners(idx, plane_models, plane_normals, plane_clouds, tol_rad)
            if fallback is not None:
                corners_per_plane[idx] = fallback.tolist()
    plane_corners = []
    for idx in range(num):
        pts = np.array(corners_per_plane[idx])
        if pts.shape != (4, 3):
            pts = find_corner_points(np.asarray(plane_clouds[idx].points), plane_models[idx])
        center = pts.mean(axis=0)
        angles = np.arctan2(*(pts - center)[:, :2].T)
        sorted_pts = pts[np.argsort(angles)]
        plane_corners.append(sorted_pts)
    return np.stack(plane_corners, axis=0)

def fill_rectangle_points(corners, num_u=50, num_v=50):
    """
    사각형 내부를 격자 점으로 채웁니다.
    
    매개변수:
        corners: (4,3) array - 사각형의 4개 모서리 점 좌표 (순서대로)
        num_u: int - u방향 분할 수 (기본값: 50)
        num_v: int - v방향 분할 수 (기본값: 50)
    
    반환값:
        points: (num_u*num_v, 3) array - 사각형 내부를 채우는 격자점들의 좌표
    """
    p00, p10, p11, p01 = corners
    us = np.linspace(0, 1, num_u)
    vs = np.linspace(0, 1, num_v)
    pts = []
    for u in us:
        for v in vs:
            p = (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + u * v * p11 + (1 - u) * v * p01
            pts.append(p)
    return np.array(pts)

def extract_and_fill(pcd, max_planes=6, min_inliers_ratio=0.03, distance_threshold=0.02, angle_tol=10.0):
    """
    포인트 클라우드에서 벽면을 추출하고 격자 점으로 채웁니다.
    
    매개변수:
        pcd: open3d.geometry.PointCloud - 입력 포인트 클라우드
        max_planes: int - 검출할 최대 평면 수 (기본값: 6)
        min_inliers_ratio: float - 최소 인라이어 비율 (기본값: 0.03)
        distance_threshold: float - RANSAC 평면 검출 거리 임계값 (기본값: 0.02)
        angle_tol: float - 평면 간 각도 허용 오차 (기본값: 10.0)
    
    반환값:
        filled: list of PointCloud - 각 벽면을 채우는 포인트 클라우드들
        vertices: (N,4,3) array - 각 벽면의 모서리 점 좌표
    """
    plane_models, plane_normals, plane_clouds = [], [], []
    remainder = pcd
    min_inliers = int(len(pcd.points)*min_inliers_ratio)
    for i in range(1, max_planes+1):
        model, normal, cloud, remainder, ok = detect_plane(remainder, distance_threshold, min_inliers)
        if not ok:
            print(f"Failed to detect plane {i}")
            break
        if not plane_models or any(abs(np.degrees(np.arccos(abs(np.dot(normal, n2))))) < angle_tol or abs(abs(np.degrees(np.arccos(abs(np.dot(normal, n2))))) - 90) < angle_tol for n2 in plane_normals):
            plane_models.append(model)
            plane_normals.append(normal)
            plane_clouds.append(cloud)
            print(f"Detected plane {i}")
    vertices = find_plane_corners_by_planes(plane_models, plane_normals, plane_clouds, angle_tol)
    filled = []
    for rect in vertices:
        pts = fill_rectangle_points(rect)
        pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        filled.append(pcl)
    return filled, vertices

def get_wall_vertices_v2(pcd, max_planes=6, min_inliers_ratio=0.03, distance_threshold=0.02, angle_tol=10.0):
    """
    포인트 클라우드에서 벽면의 꼭짓점들을 추출합니다.
    
    매개변수:
        pcd: open3d.geometry.PointCloud - 입력 포인트 클라우드
        max_planes: int - 검출할 최대 평면 수 (기본값: 6)
        min_inliers_ratio: float - 최소 인라이어 비율 (기본값: 0.03) 
        distance_threshold: float - RANSAC 평면 검출 거리 임계값 (기본값: 0.02)
        angle_tol: float - 평면 간 각도 허용 오차 (기본값: 10.0)
    
    반환값:
        wall_vertices: list - 각 벽면의 꼭짓점 좌표
        outlier_cloud: PointCloud - 평면에 속하지 않는 점들
    """
    plane_models, plane_normals, plane_clouds = [], [], []
    remainder = pcd
    min_inliers = int(len(pcd.points) * min_inliers_ratio)
    
    # 평면 검출 반복
    for i in range(1, max_planes+1):
        model, normal, cloud, remainder, ok = detect_plane(remainder, distance_threshold, min_inliers)
        if not ok:
            print(f"{i}번째 평면: 검출 실패")
            break
            
        # 이전 평면들과 각도 비교하여 수직/평행한 평면만 선택
        if not plane_models or any(abs(np.degrees(np.arccos(abs(np.dot(normal, n2))))) < angle_tol or 
                                 abs(abs(np.degrees(np.arccos(abs(np.dot(normal, n2))))) - 90) < angle_tol 
                                 for n2 in plane_normals):
            plane_models.append(model)
            plane_normals.append(normal)
            plane_clouds.append(cloud)
            print(f"{i}번째 평면: 검출 성공")
    
    # 평면들의 꼭짓점 찾기
    wall_vertices = find_plane_corners_by_planes(plane_models, plane_normals, plane_clouds, angle_tol)
    
    return wall_vertices, remainder

if __name__ == '__main__':
    input_file = "scaniverse.ply"
    pcd = o3d.io.read_point_cloud(input_file)

    wall_vertices, outlier_cloud = get_wall_vertices(pcd, max_planes=5)
    from utils import create_3d_floor_plan
    pcds = create_3d_floor_plan(wall_vertices, [])
    o3d.visualization.draw_geometries(pcds)
    o3d.visualization.draw_geometries(pcds + [pcd])
