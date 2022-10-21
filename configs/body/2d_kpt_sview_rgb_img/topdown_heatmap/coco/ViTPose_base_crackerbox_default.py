_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/ycb.py'
]
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                                    num_layers=12, 
                                    layer_decay_rate=0.75,
                                    custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            }
                                    )
                )

optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
target_type = 'GaussianHeatmap'
channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 256),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True)
)

data_cfg = dict(
    image_size=[256, 256], # change this to be smaller 
    heatmap_size=[64, 64], # 48, 64
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    det_bbox_thr=0.0,
    bbox_file='data/crackerbox/detections.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/crackerbox'
test_data_root = 'data/crackerbox_FAT'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    # val_dataloader=dict(samples_per_gpu=32),
    # test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownYCBCrackerBoxDataset',
        ann_file=f'{data_root}/keypoints.json',
        img_prefix=f'{data_root}/', # NEED TO CHANGE THIS 
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    # val=dict(
    #     type='TopDownCocoDataset',
    #     ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
    #     img_prefix=f'{data_root}/val2017/',
    #     data_cfg=data_cfg,
    #     pipeline=val_pipeline,
    #     dataset_info={{_base_.dataset_info}}),
    # test=dict(
    #     type='TopDownYCBCrackerBoxDataset',
    #     ann_file=f'{data_root}/set_3/keypoints.json',
    #     img_prefix=f'{data_root}/set_3', # NEED TO CHANGE THIS
    #     data_cfg=data_cfg,
    #     pipeline=test_pipeline,
    #     dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownYCBCrackerBoxDataset',
        ann_file=f'{test_data_root}/keypoints.json',
        img_prefix=f'{test_data_root}',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

