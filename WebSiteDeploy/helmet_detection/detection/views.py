from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadImageForm
import cv2
import sys
import os
import base64
import numpy as np
import json
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
current_folder_path = os.getcwd()
MAIN_FOLDER_PATH = os.path.dirname(os.path.dirname(current_folder_path))
sys.path.append(MAIN_FOLDER_PATH + '\\scripts_py')
from predict_helmet_demo import main

config_path = r'models\my_ssd_mobnet_v2\pipeline.config'
label_path = r'config\label_map.pbtxt'
checkpoint_path = r'models\my_ssd_mobnet_v2\checkpoint\ckpt-0'

category_index = label_map_util.create_category_index_from_labelmap(os.path.join(MAIN_FOLDER_PATH, label_path))
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(os.path.join(MAIN_FOLDER_PATH, config_path))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(MAIN_FOLDER_PATH, checkpoint_path)).expect_partial()

def detect_objects(image, category_index, detection_model):
    image_arr = np.array(image)
    img_detect = main(image_arr, category_index, detection_model)
    img_detect = cv2.cvtColor(img_detect, cv2.COLOR_RGB2BGR)
    return img_detect

def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            npimg = np.fromstring(image.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            retval, buffer = cv2.imencode('.png', img)
            uploaded_image_data = base64.b64encode(buffer).decode()

            processed_image = detect_objects(img, category_index, detection_model)
            retval, buffer = cv2.imencode('.png', processed_image)
            processed_image_data = base64.b64encode(buffer).decode()

            response_data = {
                'uploaded_image': uploaded_image_data,
                'processed_image': processed_image_data
            }
            response = HttpResponse(json.dumps(response_data), content_type="application/json")
            return response
    else:
        form = UploadImageForm()
    return render(request, 'upload_image.html', {'form': form})