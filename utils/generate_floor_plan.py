from make_wall import get_wall_vertices
from utils import create_3d_floor_plan, pcd_list_to_pcd, load_pcd, save_pcd, object_detection, create_obb_list
import open3d as o3d

def generate_floorplan_pcd_with_dbscan(pcd):
    wall_vertices, outlier_cloud = get_wall_vertices(pcd, max_planes=7)
    print(wall_vertices)
    labels = object_detection(outlier_cloud)
    obb_list = create_obb_list(outlier_cloud, labels)
    pcds = create_3d_floor_plan(wall_vertices, obb_list)
    return pcd_list_to_pcd(pcds)

def generate_floorplan_pcd_with_softgroup(pcd):
    pass    


# 사용 예시
if __name__ == "__main__":
    pcd = load_pcd("scaniverse.ply")
    pcd = generate_floorplan_pcd_with_dbscan(pcd)
    o3d.visualization.draw_geometries([pcd])
    save_pcd(pcd, "3d_floor_plan.ply")