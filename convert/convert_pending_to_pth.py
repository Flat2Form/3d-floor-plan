import os, torch, numpy as np
import open3d as o3d
import shutil

voxel_size = 0.02
pending_dir = 'convert/raw/pending'
done_dir = 'convert/raw/done'
output_dir = 'dataset/scannetv2/test'

for filename in os.listdir(pending_dir):
    if filename.endswith('.ply'):
        ply_path = os.path.join(pending_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_name = f'{base_name}_inst_nostuff.pth'
        output_path = os.path.join(output_dir, output_name)

        print(f"변환 중: {filename}")

        # 1. 로드
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # 2. voxel
        # coord = np.floor(points / voxel_size).astype(np.int32)
        # colors = colors.astype(np.float32)

        coords = np.ascontiguousarray(points - points.mean(0))
        colors = np.ascontiguousarray(colors) / 127.5 - 1

        # 3. 저장
        torch.save((coords,colors), output_path)

        # 4. 이동
        shutil.move(ply_path, os.path.join(done_dir, filename))
        print(f"→ 저장 완료: {output_path}")