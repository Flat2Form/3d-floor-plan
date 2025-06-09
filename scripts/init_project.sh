# conda create -n softgroup python=3.7
# conda activate softgroup
# 1. PyTorch + CUDA 11.7
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 2. spconv (CUDA 11.6 버전용이지만 11.7에서 잘 작동함)
pip install spconv-cu116

# 3. 기타 requirements
pip install -r requirements.txt

# 4. sparsehash 설치
apt-get update
apt-get install libsparsehash-dev -y

# 5. C++ 확장 빌드
python setup.py build_ext develop
