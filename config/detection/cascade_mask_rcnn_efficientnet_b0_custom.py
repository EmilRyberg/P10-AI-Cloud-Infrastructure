_base_ = [
    '/mmdetection/configs/_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '/mmdetection/configs/_base_/default_runtime.py'
]

checkpoint = '/checkpoints/efficientnet_224x224_custom/epoch_300.pth'  # noqa

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b0',
        drop_path_rate=0.2,
        out_indices=(2, 3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint),
    ),
    neck=dict(in_channels=[24, 40, 112, 320]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ], mask_head=dict(num_classes=2)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
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
#data = dict(train=dict(pipeline=train_pipeline))
#lr_config = dict(warmup_iters=1000, step=[27, 33])
#runner = dict(max_epochs=36)

classes = ('Rumex', 'cluster of rumex')
dataset_type = 'CocoDataset'
data_root = "/data/"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        #filter_empty_gt=False,
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
checkpoint_config = dict(interval=5)

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05,
#                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                 'relative_position_bias_table': dict(decay_mult=0.),
#                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[30, 50])
runner = dict(type='EpochBasedRunner', max_epochs=80)
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