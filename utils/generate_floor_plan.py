from make_wall import get_wall_vertices
from make_wall_v2 import get_wall_vertices_v2
from utils import create_3d_floor_plan, pcd_list_to_pcd, load_pcd, save_pcd, object_detection, create_obb_list, create_aabb_list
import open3d as o3d
from model_result import model_result_to_labels
import numpy as np

def generate_floor_plan_with_dbscan(pcd):
    wall_vertices, outlier_cloud = get_wall_vertices_v2(pcd, max_planes=6)
    labels = object_detection(outlier_cloud)
    obb_list = create_obb_list(outlier_cloud, labels)
    pcds = create_3d_floor_plan(wall_vertices, obb_list)
    return pcd_list_to_pcd(pcds)

def generate_floor_plan_with_softgroup(room_name):
    xyz, rgb, labels = model_result_to_labels(room_name)
    wall_0_idx = np.where(labels == -2)[0]
    wall_1_idx = np.where(labels == -3)[0]
    wall_0 = xyz[wall_0_idx]
    wall_1 = xyz[wall_1_idx]
    wall_0_pcd = o3d.geometry.PointCloud()
    wall_0_pcd.points = o3d.utility.Vector3dVector(wall_0)
    wall_1_pcd = o3d.geometry.PointCloud()
    wall_1_pcd.points = o3d.utility.Vector3dVector(wall_1)
    
    # 예전 방법
    # wall_vertices_0, _ = get_wall_vertices(wall_0_pcd, max_planes=5)
    # wall_vertices_1, _ = get_wall_vertices(wall_1_pcd, max_planes=1)
    # 새로운 방법
    # 두 포인트 클라우드 합치기
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(
        np.concatenate([np.asarray(wall_0_pcd.points), np.asarray(wall_1_pcd.points)], axis=0)
    )
    # 합친 포인트 클라우드에 대해 한번만 함수 적용
    wall_vertices, _ = get_wall_vertices_v2(merged_pcd, max_planes=6)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = (rgb + 1) * 127.5
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    obb_list = create_obb_list(pcd, labels)
    pcds = create_3d_floor_plan(wall_vertices_0 + wall_vertices_1, obb_list)
    return pcd_list_to_pcd(pcds)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, choices=['dbscan', 'softgroup'], required=True,
                      help='바운딩 박스 생성 방법 선택')
    parser.add_argument('--filepath', type=str, required=True, 
                      help='dbscan일 경우, 파일 경로')
    parser.add_argument('--room_name', type=str, required=True,
                      help='softgroup일 경우, 방 이름')
    parser.add_argument('--out', type=str, required=True,
                      help='출력 파일 경로')
    args = parser.parse_args()

    if args.option == 'dbscan':
        pcd = load_pcd(args.filepath)
        floor_plan_pcd = generate_floor_plan_with_dbscan(pcd)
    else:
        floor_plan_pcd = generate_floor_plan_with_softgroup(args.room_name)

    o3d.visualization.draw_geometries([floor_plan_pcd])
    
    # PLY 파일로 저장
    from bounding_box import write_ply
    points = np.asarray(floor_plan_pcd.points)
    colors = np.asarray(floor_plan_pcd.colors)
    write_ply(points, colors, None, args.out)

if __name__ == "__main__":
    main()
