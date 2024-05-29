import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np
import cv2
model_path =r'D:\Code_school_nam3ki2\KhoaHocDuLieu\NhanDienMuBaoHiem\models\my_ssd_mobnet_v4\saved_model\saved_model.pb'

config_path = r'D:\Code_school_nam3ki2\KhoaHocDuLieu\NhanDienMuBaoHiem\models\my_ssd_mobnet_v4\pipeline.config'
label_path = r'Tensorflow\workspace\annotations\label_map.pbtxt'
checkpoint_path = r'Tensorflow\workspace\models\my_ssd_mobnet'
test_folder_path = r'D:\datasets-internet - Copy\train-data\images'
result_folder_path = r'images_result'
MAIN_FOLDER_PATH = os.getcwd()

category_index = label_map_util.create_category_index_from_labelmap(os.path.join(MAIN_FOLDER_PATH, label_path))
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(os.path.join(MAIN_FOLDER_PATH, config_path))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(r'D:\Code_school_nam3ki2\KhoaHocDuLieu\NhanDienMuBaoHiem\models\my_ssd_mobnet_v4\checkpoint\ckpt-0').expect_partial()

dummy_input = tf.random.uniform((1, 320, 320, 3))

dummy_output = detection_model(dummy_input)

for var in detection_model.variables:
    print(var.name, var.shape)