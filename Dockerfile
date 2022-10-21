FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN git clone https://github.com/open-mmlab/mmcv.git \
&& cd mmcv \
&& git checkout v1.3.9 \
&& MMCV_WITH_OPS=1 pip install -e . 

RUN git clone https://github.com/andrewyguo/ViTPose.git ViTPose \
&& cd ViTPose \
&& pip install -v -e . \
&& pip install timm==0.4.9 einops \
&& pip install --no-input opencv_python==4.5.4.60 \
&& pip install numpy==1.22 \
# change branch if needed 
&& git checkout main \
&& apt-get update \
&& export DEBIAN_FRONTEND=noninteractive \ 
&& apt-get install -y libgl1 \
&& apt-get install -y unzip

# Install aws cli 
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
&& unzip awscliv2.zip \
&& ./aws/install

# Download MAE weights to /workspace
RUN wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth 

COPY ./configs /workspace/ViTPose/configs

WORKDIR /workspace/ViTPose

# Build from base container 
# FROM vitpose_base

# COPY ./configs /workspace/ViTPose/configs

# WORKDIR /workspace/ViTPose

# Run with full memory: 
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it --rm vitpose/initial_container
