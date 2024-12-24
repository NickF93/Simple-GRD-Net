import tensorflow as tf
import numpy as np

import grdnet as grd

grd.util.config_gpu()
grd.util.clear_session()
grd.util.set_seed()

image_size = 64

batch_size = 2  # Divided for total number of patches

train_valid_dataset_path    = '/tmp/ds/train'
roi_dataset_path            = '/tmp/ds/roi'
test_dataset_path           = '/tmp/ds/test'
mask_dataset_path           = '/tmp/ds/mask'
# One should use a different path for real defects and masks. This is just for testing.
real_dfct_dataset_path      = '/tmp/ds/test'
real_mask_dataset_path      = '/tmp/ds/mask'


trainer = grd.trainer.Trainer(
    directory=train_valid_dataset_path,
    roi_directory=roi_dataset_path,
    test_directory=test_dataset_path,
    mask_directory=mask_dataset_path,
    real_dfct_directory=real_dfct_dataset_path,
    real_mask_directory=real_mask_dataset_path,
    patch_size=image_size,
    patches=(1, 1),
    strides=(image_size, image_size // 2),
    channels=3,
    batch_size=batch_size,
    con_loss='huber_ssim',
    adv_loss='cos',
    use_bn=True,
    experiment_name='GRDNetTestExperiment',
)
# To cleanup tables: mysql -h localhost -u root -pBE -N -e "SET FOREIGN_KEY_CHECKS = 0; SELECT CONCAT('DROP TABLE ', table_name, ';') FROM information_schema.tables WHERE table_schema = 'GRDNetDB'" | mysql -h localhost -u root -pBE GRDNetDB -N; mysql -h localhost -u root -pBE -e "SET FOREIGN_KEY_CHECKS = 1;"
trainer.train(epochs=1000)
