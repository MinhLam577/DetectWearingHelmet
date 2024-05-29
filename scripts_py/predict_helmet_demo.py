import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import cv2

@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def predict_helmet(img_path, category_index, detection_model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    image_np_crop = image_np.copy()
    
    # Sử dụng NMS để lọc ra bounding box tốt nhất
    selected_indices = tf.image.non_max_suppression(
        detections['detection_boxes'], detections['detection_scores'], max_output_size=50)
    
    # Cập nhật lại detections
    detections['detection_boxes'] = tf.gather(detections['detection_boxes'], selected_indices).numpy()
    detections['detection_scores'] = tf.gather(detections['detection_scores'], selected_indices).numpy()
    detections['detection_classes'] = tf.gather(detections['detection_classes'], selected_indices).numpy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'], 
                detections['detection_classes'] + label_id_offset, #Cộng vô để khớp với category_index
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True, #Chuấn hóa về 0 => 1
                max_boxes_to_draw=100, 
                min_score_thresh = 0.5,
                agnostic_mode=False) #Tất cả các hộp đều được vẽ cùng màu
    return image_np_with_detections


def predict_helmet_2(img, category_index, detection_model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    image_np_crop = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'], 
                detections['detection_classes'] + label_id_offset, #Cộng vô để khớp với category_index
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True, #Chuấn hóa về 0 => 1
                max_boxes_to_draw=100, 
                min_score_thresh = 0.5,
                agnostic_mode=False) #Tất cả các hộp đều được vẽ cùng màu
    return image_np_with_detections
        
def Get_ground_truth_from_xml(annotation_path, category_index):
    import xml.etree.ElementTree as ET
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    names = []
    for member in root.findall('object'):
        names.append(member[0].text)
        box = []
        box.append(member[5][0].text) #xmin
        box.append(member[5][1].text) #ymin
        box.append(member[5][2].text) #xmax
        box.append(member[5][3].text) #ymax
        boxes.append(box)
    classes = [v.get('id') for n in names for k, v in category_index.items() if v.get('name') == n]
    return [list(map(int, box)) for box in boxes], classes, names

def draw_ground_truth(img_path, boxes, classes):
    img = cv2.imread(img_path)
    ground_truth_boxes = boxes
    ground_truth_classes = classes
    for i in range(len(ground_truth_boxes)):
        if ground_truth_classes[i] == 'helmet':
            cv2.rectangle(img, (ground_truth_boxes[i][0], ground_truth_boxes[i][1]), (ground_truth_boxes[i][2], ground_truth_boxes[i][3]), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (ground_truth_boxes[i][0], ground_truth_boxes[i][1]), (ground_truth_boxes[i][2], ground_truth_boxes[i][3]), (0, 0, 255), 2)
    return img

def main(img, category_index, detection_model):
    img = predict_helmet_2(img, category_index, detection_model)
    return img
