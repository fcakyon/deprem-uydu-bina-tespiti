_base_ = '../solov2/solov2_r50_fpn_3x_coco.py'

NUM_CLASSES = 1
CLASSES = ('building',)
LOAD_FROM = 'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth'
DATA_ROOT = 'data/xview2_pre/'
TRAIN_JSON_PATH = 'train.json'
TRAIN_IMG_FOLDER = 'train/images'
VAL_JSON_PATH = 'val.json'
VAL_IMG_FOLDER = 'val/images'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800), (1333, 768), (1333, 736), (1333, 704),
                   (1333, 672), (1333, 640)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

data_root = DATA_ROOT
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=CLASSES,
        ann_file=data_root + TRAIN_JSON_PATH,
        img_prefix=data_root + TRAIN_IMG_FOLDER,
        pipeline=train_pipeline),
    val=dict(
        classes=CLASSES,
        ann_file=data_root + VAL_JSON_PATH,
        img_prefix=data_root + VAL_IMG_FOLDER,
        pipeline=test_pipeline),
    test=dict(
        classes=CLASSES,
        ann_file=data_root + VAL_JSON_PATH,
        img_prefix=data_root + VAL_IMG_FOLDER,
        pipeline=test_pipeline))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# runtime settings
workflow = [('train', 1), ('val', 1)]
auto_scale_lr = dict(enable=True, base_batch_size=16)
load_from = LOAD_FROM
checkpoint_config = dict(interval=1, save_optimizer=False)
evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best="auto")

# model settings
model = dict(
    type='SOLOv2',
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=NUM_CLASSES,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=300))