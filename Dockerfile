## cudaのイメージをインポート (https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=12.1)
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
ARG TORCH=2.2.0

# タイムゾーンの指定。これがないとビルドの途中でCUIインタラクションが発生し停止する
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

## install apt packages.
RUN apt-get update && \
    apt-get install -y \
    wget \
    bzip2 \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.8 \
    python3-pip

# キャッシュの削除。これをすることで多少imageが軽くなる。
RUN rm -rf /var/lib/apt/lists/*

# python パッケージのインストール
RUN pip3 install --no-cache-dir matplotlib
RUN pip3 install --no-cache-dir plotly==5.9.0
RUN pip3 install --no-cache-dir networkx==2.8.8
RUN pip3 install --no-cache-dir \
    torch==${TORCH} torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir torchmetrics==0.11.4
RUN pip3 install --no-cache-dir \
    pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
RUN pip3 install --no-cache-dir torch_geometric==2.3.0
RUN pip3 install --no-cache-dir plyfile
RUN pip3 install --no-cache-dir h5py
RUN pip3 install --no-cache-dir colorhash
RUN pip3 install --no-cache-dir seaborn
RUN pip3 install --no-cache-dir numba
RUN pip3 install --no-cache-dir pytorch-lightning
RUN pip3 install --no-cache-dir pyrootutils
RUN pip3 install --no-cache-dir hydra-core --upgrade
RUN pip3 install --no-cache-dir hydra-colorlog
RUN pip3 install --no-cache-dir hydra-submitit-launcher
RUN pip3 install --no-cache-dir rich
RUN pip3 install --no-cache-dir torch_tb_profiler
RUN pip3 install --no-cache-dir wandb
RUN pip3 install --no-cache-dir open3d
RUN pip3 install --no-cache-dir gdown
# RUN python3 -m pip install pgeof
WORKDIR /point_segmentation_models