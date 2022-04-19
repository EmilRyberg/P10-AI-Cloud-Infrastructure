_base_ = [
    "/mmclassification/configs/_base_/datasets/pipelines/rand_aug.py",
    "/mmclassification/configs/_base_/models/efficientnet_b0.py",
    "/mmclassification/configs/_base_/default_runtime.py"
]

model = dict(
    head=dict(
        _delete_=True,
        type='MultiLabelLinearClsHead',
        num_classes=1,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True)),
    train_cfg=None
      #train_cfg=dict(augments=[
      #    dict(type='BatchMixup', alpha=0.8, num_classes=2, prob=0.5),
      #    dict(type='BatchCutMix', alpha=1.0, num_classes=2, prob=0.5)
      #])
)
classes = None #("bg", "Rumex")
dataset_type = 'CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.1,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 6,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='Resize', size=(256, -1)),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        type=dataset_type,
        data_prefix='data/train',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        data_prefix='data/val',
        ann_file='data/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type=dataset_type,
        data_prefix='data/test',
        ann_file='data/meta/test.txt',
        pipeline=test_pipeline))
#evaluation = dict(interval=1, metric='accuracy', metric_options=dict(
#    topk=(1,)
#))
evaluation=dict(interval=1, metric='binary_accuracy')

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

optimizer = dict(
    type='AdamW',
    lr=0.01,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
checkpoint_config = dict(interval=20)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=300)