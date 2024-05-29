import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

TRAINING_SCRIPT = TRAINING_SCRIPT = os.path.join(MAIN_FOLDER_PATH, r'scripts_py\model_main_tf2.py')

checkpoint_path = paths["CHECKPOINT_PATH"]
savepoint_path = paths["CHECKPOINT_PATH"]

command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={} --alsologtostderr".format(TRAINING_SCRIPT, savepoint_path , files['PIPELINE_CONFIG'], checkpoint_path)
os.system(f"{command}")