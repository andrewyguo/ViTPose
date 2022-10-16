FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN git clone https://github.com/open-mmlab/mmcv.git \
&& cd mmcv \
&& git checkout v1.3.9 \
&& MMCV_WITH_OPS=1 pip install -e . 

RUN git clone https://github.com/ViTAE-Transformer/ViTPose.git ViTPose \
&& cd ViTPose \
&& pip install -v -e . \
&& pip install timm==0.4.9 einops \
&& pip install --no-input opencv_python==4.5.4.60 \
&& pip install numpy==1.22 \
&& apt-get update \
&& export DEBIAN_FRONTEND=noninteractive \ 
&& apt-get install -y libgl1 \
&& apt-get install -y s3cmd 

# Download MAE weights to /workspace
RUN wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth 

WORKDIR /workspace/ViTPose

# Run with full memory: 
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it --rm vitpose/initial_container

# Run training 
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py 

# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py 1 --cfg-options model.pretrained=/workspace/mae_pretrain_vit_base.pth --seed 0


# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py 1 --cfg-options model.pretrained=venv/mae_pretrain_vit_base.pth --seed 0

# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_crackerbox.py 1 --cfg-options model.pretrained=venv/mae_pretrain_vit_base.pth --seed 0

# OLD RUN COMMANDS
# RUN pip install -r requirements.txt \
# && pip install --no-input opencv_python==4.5.4.60 \
# && pip install -r requirements/mminstall.txt \
# && pip install  --no-input mmpose \
# && apt-get update \
# && export DEBIAN_FRONTEND=noninteractive \ 
# && apt-get install -y libgl1 \
# # Older versions of numpy were causing issues 
# && pip uninstall --no-input numpy \ 
# && pip install numpy==1.22 \
# && pip install -e -v .