import open3d as o3d
import numpy as np

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
    o3d.visualization.draw_geometries(obb_list)