_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/s3dis.py',  # 更新为引用S3DIS数据集
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (768, 1024)  # S3DIS图像大小
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

# 重新配置数据采样器。
train_dataloader = dict(batch_size=4, num_workers=4)  # 根据需要调整批量大小
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# 重新配置优化器。
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
