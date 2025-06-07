"""Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py."""

import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil

import numpy as np
import plyfile
import torch

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='test')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))

# data 폴더에서 모든 파일 가져오기
files = sorted(glob.glob('data/*.ply'))

# done 폴더가 없으면 생성
os.makedirs('data/done', exist_ok=True)
# test 폴더가 없으면 생성
os.makedirs(split, exist_ok=True)

def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    # test 폴더에 저장
    output_file = os.path.join(split, os.path.basename(fn)[:-4] + '_inst_nostuff.pth')
    torch.save((coords, colors), output_file)
    print(f'Saved to {output_file}')

    # 원본 파일을 done 폴더로 이동
    done_file = os.path.join('data/done', os.path.basename(fn))
    shutil.move(fn, done_file)
    print(f'Moved to {done_file}')

p = mp.Pool(processes=mp.cpu_count())
p.map(f_test, files)
p.close()
p.join()
