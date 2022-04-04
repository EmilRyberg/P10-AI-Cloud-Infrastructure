_base_ = [
    '/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
]

classes = ('rumex with leaves', 'rumex with leaves IS', 'rumex stalks only', 'cluster of rumex', 'ignore', 'rumex_generated_med_conf', 'rumex_generated_2_med_conf', 'rumex_generated_high_conf', 'rumex_generated_2_high_conf')
data = dict(train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
optimizer_config = None
model = dict(
    roi_head=dict(
        bbox_head=dict(type='Shared2FCBBoxHead', num_classes=9),
        mask_head=dict(num_classes=9)
    )
)


# do not use mmdet version fp16
#fp16 = None
#optimizer_config = dict(
#    type="DistOptimizerHook",
#    update_interval=1,
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1,    use_fp16=True,
#)