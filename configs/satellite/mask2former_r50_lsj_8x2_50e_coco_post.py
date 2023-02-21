_base_ = ['../mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py']
NEPTUNE_API_TOKEN = ''
NEPTUNE_PROJECT = 'OBSS-ML/deprem-building-damage-detection'
EXPERIMENT_NAME = 'experiment_name'
LOAD_FROM = 'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

WORK_DIR = 'experiments/' + EXPERIMENT_NAME

DATASET_TYPE = 'CocoDataset'
DATA_ROOT = 'data/xview2_post/'
TRAIN_JSON_PATH = 'train.json'
TRAIN_IMG_FOLDER = 'train/images'
VAL_JSON_PATH = 'val.json'
VAL_IMG_FOLDER = 'val/images'

IMAGE_SIZE = (512, 512)
CLASSES = ('no-damage', 'minor-damage', 'major-damage', 'destroyed', )
NUM_THING_CLASSES = 4
NUM_STUFF_CLASSES = 0
NUM_CLASSES = NUM_THING_CLASSES + NUM_STUFF_CLASSES

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=IMAGE_SIZE,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=IMAGE_SIZE,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='Pad', size=IMAGE_SIZE, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=IMAGE_SIZE,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = DATASET_TYPE
data_root = DATA_ROOT
data = dict(
    _delete_=True,
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + TRAIN_JSON_PATH,
        img_prefix=data_root + TRAIN_IMG_FOLDER,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + VAL_JSON_PATH,
        img_prefix=data_root + VAL_IMG_FOLDER,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + VAL_JSON_PATH,
        img_prefix=data_root + VAL_IMG_FOLDER,
        pipeline=test_pipeline))


# logger settings
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook',
             reset_flag=False),
		dict(
            type='NeptuneLoggerHook',
            init_kwargs=dict(
                project=NEPTUNE_PROJECT,
                name=EXPERIMENT_NAME,
                api_token=NEPTUNE_API_TOKEN))
    ])


# runtime settings
load_from = LOAD_FROM
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=24)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=True,
    step=[20, 22],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)
workflow = [('train', 1), ('val', 1)]
auto_scale_lr = dict(enable=True, base_batch_size=16)
checkpoint_config = dict(
    by_epoch=True, interval=1, save_last=True, max_keep_ckpts=3)
evaluation = dict(_delete_=True,interval=1, metric=['bbox', 'segm'])
work_dir = WORK_DIR


# model setting
model = dict(
    panoptic_head=dict(
        num_things_classes=NUM_THING_CLASSES,
        num_stuff_classes=NUM_STUFF_CLASSES,
        loss_cls=dict(class_weight=[1.0] * NUM_CLASSES + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=NUM_THING_CLASSES,
        num_stuff_classes=NUM_STUFF_CLASSES),
    test_cfg=dict(panoptic_on=False))