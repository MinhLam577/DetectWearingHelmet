from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadImageForm
import cv2
import sys
import os
import base64
import numpy as np
import json
current_folder_path = os.getcwd()
MAIN_FOLDER_PATH = os.path.dirname(os.path.dirname(current_folder_path))
sys.path.append(MAIN_FOLDER_PATH)
from predict_helmet_demo import main

def detect_objects(image):
    image_arr = np.array(image)
    img_detect = main(image_arr)
    img_detect = cv2.cvtColor(img_detect, cv2.COLOR_RGB2BGR)
    img_detect = cv2.resize(img_detect, (800, 600))
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

            processed_image = detect_objects(img)
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