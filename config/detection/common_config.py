_base_ = [
    '/mmdetection/configs/_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
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
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('Rumex',)
dataset_type = 'CocoDataset'
data_root = "/data/"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
checkpoint_config = dict(interval=20)

optimizer = dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
#optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)                                              'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=dict(max_norm=35.0))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[105, 115])
runner = dict(type='EpochBasedRunner', max_epochs=120)

#optimizer_config = None
#model = dict(
#    roi_head=dict(
#        bbox_head=dict(type='Shared2FCBBoxHead', num_classes=9),
#        mask_head=dict(num_classes=9)
#    )
#)


# do not use mmdet version fp16
#fp16 = None
#optimizer_config = dict(
#    type="DistOptimizerHook",
#    update_interval=1,
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1,    use_fp16=True,
#)