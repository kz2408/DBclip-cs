# model settings
model = dict(
    type='SimpleClassifier',
    backbone=dict(
        type='CLIPVisionTransformerEnhanced',  # 使用增强版CLIP ViT
        clip_model="ViT-B/16",
        img_size=224,
        frozen_stages=0),  # 设置为0，允许整个CLIP模型微调
    neck=dict(
        type='MultiScalePFC',
        in_channels_list=[256, 512, 1024],  # ViT-B/16特征的一半、原始和两倍通道数
        out_channels=256,
        dropout=0.5),
    head=dict(
        type='ClsHead',
        in_channels=256,
        num_classes=20,
        method='fc',
        loss_cls=dict(
            type='LabelSmoothingLoss', use_sigmoid=True,
            smoothing=0.1,
            loss_weight=1.0)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = 'VOCDataset'
data_root = './data/voc2012/VOCdevkit/'
online_data_root = 'appendix/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
extra_aug = dict(
    photo_metric_distortion=dict(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    random_crop=dict(
        min_crop_size=0.8
    )
)
data = dict(
    imgs_per_gpu=12,  # 减小batch size因为模型更大
    workers_per_gpu=2,
    sampler='ClassAware',
    train=dict(
            type=dataset_type,
            ann_file=online_data_root + 'longtail2012/img_id.txt',
            img_prefix=data_root + 'VOC2012/',
            img_scale=(224, 224),
            img_norm_cfg=img_norm_cfg,
            extra_aug=extra_aug,
            size_divisor=32,
            resize_keep_ratio=False,
            flip_ratio=0.5,
            use_splicemix=True
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        class_split=online_data_root + 'longtail2012/class_split.pkl',
        img_scale=(224, 224),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        resize_keep_ratio=False,
        flip_ratio=0))
# optimizer
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.05)  # 使用更小的学习率以微调
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[5, 7])
checkpoint_config = dict(interval=4)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=2)
# runtime settings
total_epochs = 20  # 减少训练轮数，因为使用的是强大的预训练模型
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/LT_vit_clip_enhanced_pfc_DB'
load_from = None
resume_from = None
workflow = [('train', 1)]