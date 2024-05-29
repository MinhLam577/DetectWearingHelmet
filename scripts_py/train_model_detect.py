import os
from object_detection.utils import config_util
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
MAIN_FOLDER_PATH = os.getcwd()

paths = {
    'WORKSPACE_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join(MAIN_FOLDER_PATH,'Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(MAIN_FOLDER_PATH,'Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'EXPORTED_MODEL_PATH': os.path.join(MAIN_FOLDER_PATH,'Tensorflow', 'workspace','exported-models', CUSTOM_MODEL_NAME),
    'PROTOC_PATH':os.path.join(MAIN_FOLDER_PATH,'Tensorflow','protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join(MAIN_FOLDER_PATH,'Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Create Directory Structure
for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path)

# Create Label Map
labels = [
    {'name':'head', 'id':1},
    {'name':'helmet', 'id':2},
]   
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
        
# img_train_dir = r"D:\Code_school_nam3ki2\KhoaHocDuLieu\NhanDienMuBaoHiem\Tensorflow\workspace\images\Dataset_1_Test\train"
# img_valid_dir = r"D:\Code_school_nam3ki2\KhoaHocDuLieu\NhanDienMuBaoHiem\Tensorflow\workspace\images\Dataset_1_Test\valid"
# img_test_dir = r"D:\Code_school_nam3ki2\KhoaHocDuLieu\NhanDienMuBaoHiem\Tensorflow\workspace\images\Dataset_1_Test\test"


# # Create TF recordsk
# if not os.path.exists(files['TF_RECORD_SCRIPT']):
#     os.system(f"git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}")
# os.system(f"python {files['TF_RECORD_SCRIPT']} -x {img_train_dir} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}")
# os.system(f"python {files['TF_RECORD_SCRIPT']} -x {img_valid_dir} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'valid.record')}")
# os.system(f"python {files['TF_RECORD_SCRIPT']} -x {img_test_dir} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}")

num_steps = 100000
warm_step = 4000
learning_rate_base = 0.15
warmup_learning_rate = 0.05
total_steps = num_steps + warm_step

# #Copy model config từ file config của pretrain_model
# command_copy_model_config = f"copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}"
# os.system(command_copy_model_config)

# Đọc tệp pipeline.config và lấy ra tất cả các cấu hình.
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

# Custom lại model ssds
configs['model'].ssd.num_classes = len(labels)
configs['model'].ssd.post_processing.batch_non_max_suppression.max_detections_per_class = 20
configs['model'].ssd.post_processing.batch_non_max_suppression.max_total_detections = 40
configs['model'].ssd.image_resizer.fixed_shape_resizer.height = 320
configs['model'].ssd.image_resizer.fixed_shape_resizer.width = 320
configs['train_config'].batch_size = 4
configs['train_config'].fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
configs['train_config'].fine_tune_checkpoint_type = "detection"
configs['train_config'].num_steps = num_steps
configs['train_config'].optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = total_steps
configs['train_config'].optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = warmup_learning_rate
configs['train_config'].optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = learning_rate_base
configs['train_config'].optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = warm_step
configs['train_config'].max_number_of_boxes = 40
configs['train_input_config'].label_map_path = files['LABELMAP']
configs['train_input_config'].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
configs['eval_input_configs'][0].label_map_path = files['LABELMAP']
configs['eval_input_configs'][0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'valid.record')]

# Lưu lại custom model vào file config
pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
config_util.save_pipeline_config(pipeline_config, os.path.dirname(files['PIPELINE_CONFIG']))

save_dir = paths['CHECKPOINT_PATH']

#Train model
TRAINING_SCRIPT = r'scripts_py'
command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={} --alsologtostderr".format(TRAINING_SCRIPT, save_dir, files['PIPELINE_CONFIG'], num_steps)

os.system(command)
