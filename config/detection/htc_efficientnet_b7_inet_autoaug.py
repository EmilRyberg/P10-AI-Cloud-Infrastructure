_base_ = [
    "htc_resnet50_inet_autoaug.py"
]

checkpoint = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth"

model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        out_indices=(2, 3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint),
    ),
    neck=dict(in_channels=[48, 80, 224, 640]),
    roi_head=dict(
        semantic_head=dict(
            fusion_level = 1
        )
    )
)