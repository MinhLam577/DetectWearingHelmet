import xml.etree.cElementTree as ET
import cv2
import os
import re

# Read image
def ReadImage(filename):
    img = cv2.imread(filename)
    return img

#Read label
def ReadLabel(label_path):
    with open(label_path, 'r') as file:
        data = file.read()
        pattern = r"item\s*{\s*name:'(.*?)'\s*id:(\d+)\s*}"
        result = re.findall(pattern, data, re.MULTILINE)
        return [{"name": name, "id": int(id) - 1} for name, id in result]

def ReadAnnotation(filename):
    annotation = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            data = [float(i) for i in data]
            annotation.append(data)
    return annotation

def convert_yolo_to_xml(yolo_data_path, img_path, label_path, save_dir):
    label_data = ReadLabel(label_path)
    yolo_data_arr = ReadAnnotation(yolo_data_path)
    img = ReadImage(img_path)
    img_height,img_width, _ = img.shape
    img_name = img_path.split('\\')[-1]
    
    # Create XML tree structure
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = img_name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"
    for i in range(len(yolo_data_arr)):
        yolo_data = yolo_data_arr[i]
        id = int(yolo_data[0])
        # Convert YOLO center coordinates (normalized) to XML format
        xmin = (yolo_data[1] - yolo_data[3]/2.) * img_width
        xmax = (yolo_data[1] + yolo_data[3]/2.) * img_width
        ymin = (yolo_data[2] - yolo_data[4]/2.) * img_height
        ymax = (yolo_data[2] + yolo_data[4]/2.) * img_height
        
        # Create XML tree structure
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label_data[id].get('name')
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))

    # Convert to string
    xml_str = ET.tostring(annotation)

    # Save to file
    save_path = os.path.join(save_dir, img_name.split('.')[0] + ".xml")
    with open(save_path, "wb") as f:
        f.write(xml_str)