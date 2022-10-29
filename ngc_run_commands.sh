export CONFIG=ViTPose_large_crackerbox_200e
export CONFIG=ViTPose_base_crackerbox_default
export NUM_GPU=1
# Setup aws cli credentials

rm ~/.aws/config
mkdir ~/.aws
echo "[default]" >> ~/.aws/config
echo "aws_access_key_id = andrewg" >> ~/.aws/config
echo "aws_secret_access_key = d11324b6fa5020a7a35b34cf1247db95" >> ~/.aws/config 
echo "region = us-east-1" >> ~/.aws/config 
echo "s3 = " >> ~/.aws/config 
echo "    endpoint_url = https://pbss.s8k.io" >> ~/.aws/config 
echo "    signature_version = s3v4" >> ~/.aws/config 
echo "    payload_signing_enabled = true" >> ~/.aws/config 
echo "max_attempts = 10" >> ~/.aws/config 
echo "" >> ~/.aws/config 

# Get data
# aws s3 cp s3://crackerbox_zip/crackerbox_1.zip ./data/crackerbox/crackerbox_1.zip --endpoint-url https://pbss.s8k.io
# aws s3 cp s3://crackerbox_zip/crackerbox_2.zip ./data/crackerbox/crackerbox_2.zip --endpoint-url https://pbss.s8k.io
# aws s3 cp s3://crackerbox_zip/crackerbox_3.zip ./data/crackerbox/crackerbox_3.zip --endpoint-url https://pbss.s8k.io
# aws s3 cp s3://crackerbox_zip/crackerbox_4.zip ./data/crackerbox/crackerbox_4.zip --endpoint-url https://pbss.s8k.io
# aws s3 cp s3://crackerbox_zip/crackerbox_5.zip ./data/crackerbox/crackerbox_5.zip --endpoint-url https://pbss.s8k.io
# aws s3 cp s3://crackerbox_zip/crackerbox_6.zip ./data/crackerbox/crackerbox_6.zip --endpoint-url https://pbss.s8k.io
mkdir -p ./data/crackerbox
aws s3 sync s3://crackerbox_zip ./data/crackerbox --endpoint-url https://pbss.s8k.io
aws s3 sync s3://crackerbox_zip ./data/crackerbox --endpoint-url https://pbss.s8k.io
aws s3 sync s3://crackerbox_zip ./data/crackerbox --endpoint-url https://pbss.s8k.io
aws s3 sync s3://crackerbox_zip ./data/crackerbox --endpoint-url https://pbss.s8k.io
aws s3 sync s3://crackerbox_zip ./data/crackerbox --endpoint-url https://pbss.s8k.io
aws s3 sync s3://crackerbox_zip ./data/crackerbox --endpoint-url https://pbss.s8k.io

unzip -q ./data/crackerbox/crackerbox_1.zip -d ./data/crackerbox
unzip -q ./data/crackerbox/crackerbox_2.zip -d ./data/crackerbox
unzip -q ./data/crackerbox/crackerbox_3.zip -d ./data/crackerbox
unzip -q ./data/crackerbox/crackerbox_4.zip -d ./data/crackerbox
unzip -q ./data/crackerbox/crackerbox_5.zip -d ./data/crackerbox
unzip -q ./data/crackerbox/crackerbox_6.zip -d ./data/crackerbox

# Combine data into one keypoints.json file 
python convert_dope_to_coco.py --data data/crackerbox --outf data/crackerbox

# Start training 
bash tools/dist_train.sh \
configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/$CONFIG.py \
$NUM_GPU --cfg-options model.pretrained=/workspace/mae_pretrain_vit_base.pth \
data.samples_per_gpu=10 \
--seed 0

# Save weights to s3 
aws s3 mb s3://$CONFIG --endpoint-url https://pbss.s8k.io

aws s3 sync work_dirs s3://$CONFIG --endpoint-url https://pbss.s8k.io
aws s3 sync work_dirs s3://$CONFIG --endpoint-url https://pbss.s8k.io
aws s3 sync work_dirs s3://$CONFIG --endpoint-url https://pbss.s8k.io
aws s3 sync work_dirs s3://$CONFIG --endpoint-url https://pbss.s8k.io
aws s3 sync work_dirs s3://$CONFIG --endpoint-url https://pbss.s8k.io
aws s3 sync work_dirs s3://$CONFIG --endpoint-url https://pbss.s8k.io

# Run training 
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py 

# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_crackerbox_test.py 1 --cfg-options model.pretrained=venv/mae_pretrain_vit_base.pth --seed 0


# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py 1 --cfg-options model.pretrained=venv/mae_pretrain_vit_base.pth --seed 0

# bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_crackerbox.py 1 --cfg-options model.pretrained=venv/mae_pretrain_vit_base.pth --seed 0
# --cfg-options model.pretrained=venv/mae_pretrain_vit_large.pth --seed 0

# bash tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_crackerbox.py work_dirs/ViTPose_base_crackerbox/epoch_210.pth 1

# Base Config 
bash tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_crackerbox_test.py \
/disk2/vitpose_weights/ViTPose_base_crackerbox_400e/epoch_400.pth 1

# Large 
bash tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_crackerbox_200e.py \
/disk2/vitpose_weights/ViTPose_large_crackerbox_200e/epoch_120.pth 1 \
--cfg-options \
