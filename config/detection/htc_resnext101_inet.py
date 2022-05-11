_base_ = [
    "htc_resnet50_inet.py"
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d'),
    ),
    neck=dict(in_channels=[256, 512, 1024, 2048])
)