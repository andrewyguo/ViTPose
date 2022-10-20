# Setup aws cli credentials
mkdir ~/.aws
echo "[default]" >> ~/.aws/config
echo "aws_access_key_id = andrewg" >> ~/.aws/config
echo "aws_secret_access_key = d11324b6fa5020a7a35b34cf1247db95" >> ~/.aws/config 
echo "region = us-east-1" >> ~/.aws/config 
echo "s3 = " >> ~/.aws/config 
echo "    endpoint_url = https://pbss.s8k.io" >> ~/.aws/config 
echo "    signature_version = s3v4" >> ~/.aws/config 
echo "    payload_signing_enabled = true" >> ~/.aws/config 
echo "    max_concurrent_requests = 10" >> ~/.aws/config 
echo "" >> ~/.aws/config 

# Get data
aws s3 cp s3://coco_crackerbox_60K/crackerbox_data.zip ./data/crackerbox_data.zip --endpoint-url https://pbss.s8k.io
unzip -q ./data/crackerbox_data 

# Start training 
bash tools/dist_train.sh \
configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_crackerbox_default.py \
1 --cfg-options model.pretrained=/workspace/mae_pretrain_vit_base.pth --seed 0

# Save weights to s3 
aws s3 mb s3://ViTPose_base_crackerbox_default --endpoint-url https://pbss.s8k.io
aws s3 cp work_dirs s3://ViTPose_base_crackerbox_default --recursive --endpoint-url https://pbss.s8k.io

# Run training 
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py 

# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py 1 --cfg-options model.pretrained=/workspace/mae_pretrain_vit_base.pth --seed 0


# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py 1 --cfg-options model.pretrained=venv/mae_pretrain_vit_base.pth --seed 0

# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_crackerbox.py 1 --cfg-options model.pretrained=venv/mae_pretrain_vit_base.pth --seed 0


# bash tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_crackerbox.py work_dirs/ViTPose_base_crackerbox/epoch_210.pth 1

