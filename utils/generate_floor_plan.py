from make_wall import get_wall_vertices
from utils import create_3d_floor_plan, pcd_list_to_pcd, load_pcd, save_pcd, object_detection, create_obb_list, create_aabb_list
import open3d as o3d
from model_result import model_result_to_labels
import numpy as np

def generate_floorplan_with_dbscan(pcd):
    wall_vertices, outlier_cloud = get_wall_vertices(pcd, max_planes=5)
    print(wall_vertices)
    labels = object_detection(outlier_cloud)
    obb_list = create_obb_list(outlier_cloud, labels)
    pcds = create_3d_floor_plan(wall_vertices, obb_list)
    return pcd_list_to_pcd(pcds)

def generate_floorplan_with_softgroup(room_name):
    xyz, rgb, labels = model_result_to_labels(room_name)
    wall_0_idx = np.where(labels == -2)[0]
    wall_1_idx = np.where(labels == -3)[0]
    wall_0 = xyz[wall_0_idx]
    wall_1 = xyz[wall_1_idx]
    wall_0_pcd = o3d.geometry.PointCloud()
    wall_0_pcd.points = o3d.utility.Vector3dVector(wall_0)
    wall_1_pcd = o3d.geometry.PointCloud()
    wall_1_pcd.points = o3d.utility.Vector3dVector(wall_1)
    wall_vertices_0, _ = get_wall_vertices(wall_0_pcd, max_planes=5)
    wall_vertices_1, _ = get_wall_vertices(wall_1_pcd, max_planes=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = (rgb + 1) * 127.5
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print(wall_vertices_0)
    obb_list = create_obb_list(pcd, labels)
    pcds = create_3d_floor_plan(wall_vertices_0 + wall_vertices_1, obb_list)
    return pcd_list_to_pcd(pcds)

# 사용 예시
if __name__ == "__main__":
    # pcd = load_pcd("utils/scaniverse.ply")
    floorplan_pcd = generate_floorplan_with_softgroup("scaniverse-model")
    o3d.visualization.draw_geometries([floorplan_pcd])
    from bounding_box import write_ply
    # write_ply(pcd.points, pcd.colors, None, "3d_floor_plan.ply")
    # wall_vertices, outlier_cloud = get_wall_vertices(pcd, max_planes=6)
    # print(wall_vertices)
    # labels = object_detection(outlier_cloud)
    # obb_list = create_aabb_list(outlier_cloud, labels)
    # pcds = create_3d_floor_plan(wall_vertices, [aabb.get_oriented_bounding_box() for aabb in obb_list])
    # ret = pcd_list_to_pcd(pcds)
    # o3d.visualization.draw_geometries([ret])
    # o3d.visualization.draw_geometries([ret] + [outlier_cloud])
    # o3d.visualization.draw_geometries([outlier_cloud])