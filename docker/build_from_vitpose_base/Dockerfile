FROM vitpose_base

COPY ./configs /workspace/ViTPose/configs

WORKDIR /workspace/ViTPose

# Run with full memory: 
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it --rm vitpose/initial_container
