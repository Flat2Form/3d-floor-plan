import open3d as o3d
import numpy as np

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

def create_xy_aligned_bounding_boxes(pcd, instance_labels):
    coords = np.asarray(pcd.points)
    obb_list = []
    for label in np.unique(instance_labels):
        if label < 0:
            continue
        mask = instance_labels == label
        label_coords = coords[mask]
        if len(label_coords) == 0:
            continue

        # 1. x, y만 추출해서 2D OBB 구하기
        xy = label_coords[:, :2]
        # Open3D의 2D OBB는 없으므로, numpy로 최소직사각형을 구함
        # SVD로 2D OBB 구하기
        center = xy.mean(axis=0)
        cov = np.cov(xy - center, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        rot = eigvecs

        # 회전된 좌표계로 변환
        xy_rot = (xy - center) @ rot
        min_xy = xy_rot.min(axis=0)
        max_xy = xy_rot.max(axis=0)

        # 2. z축 min/max 구하기
        min_z = label_coords[:, 2].min()
        max_z = label_coords[:, 2].max()

        # 3. 8개 꼭짓점 생성 (xy평면에서 OBB, z축은 min/max)
        corners = np.array([
            [min_xy[0], min_xy[1], min_z],
            [max_xy[0], min_xy[1], min_z],
            [max_xy[0], max_xy[1], min_z],
            [min_xy[0], max_xy[1], min_z],
            [min_xy[0], min_xy[1], max_z],
            [max_xy[0], min_xy[1], max_z],
            [max_xy[0], max_xy[1], max_z],
            [min_xy[0], max_xy[1], max_z],
        ])
        # 다시 원래 좌표계로 변환
        corners[:, :2] = corners[:, :2] @ rot.T + center

        # 4. Open3D OBB 생성
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
        aabb.color = (0, 0, 1)
        obb_list.append(aabb)
    return obb_list

def obb_to_pcd(obb, num_points_per_edge=100):
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    verts = np.asarray(line_set.points)
    edges = np.asarray(line_set.lines)

    sampled_pts = []
    for (i, j) in edges:
        p1, p2 = verts[i], verts[j]
        for k in range(num_points_per_edge + 1):
            t = k / num_points_per_edge
            sampled_pts.append((1 - t) * p1 + t * p2)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(sampled_pts))
    pcd.paint_uniform_color((1, 0, 0))
    return pcd

def create_bounding_boxes(pcd, instance_labels):
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="PLY 파일 경로")
    parser.add_argument("--label_path", type=str, required=True, help="레이블 파일 경로")
    args = parser.parse_args()
    
    # PLY 파일 로드
    pcd = o3d.io.read_point_cloud(args.ply_path)
    
    # 레이블 파일 로드
    labels = np.load(args.label_path)
    
    # 바운딩 박스 생성
    obb_list = create_bounding_boxes(pcd, labels)
    # o3d.visualization.draw_geometries(obb_list)
    obb_pcd_list = [obb_to_pcd(obb) for obb in obb_list]
    o3d.visualization.draw_geometries(obb_pcd_list)
    # xy_aabb_list = create_xy_aligned_bounding_boxes(pcd, labels)
    # o3d.visualization.draw_geometries(xy_aabb_list)
    # aabb_list = create_aabb_list(pcd, labels)
    # o3d.visualization.draw_geometries(aabb_list)