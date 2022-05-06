_base_ = [
    "htc_resnet50_inet.py"
]

checkpoint = "https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth"

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], # these values can be found in the code for Swin transformer in mmdetection or mmclassification
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=False,
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint),
    ),
    neck=dict(in_channels=[128, 256, 512, 1024])
)

optimizer = dict(lr=0.0002)