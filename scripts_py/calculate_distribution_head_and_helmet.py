import xml.etree.ElementTree as ET
import os
def count_tags_head_and_helmet(file_path):
    # Parse XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    count_head = 0
    count_helmet = 0
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'head':
            count_head += 1
        elif name == 'helmet':
            count_helmet += 1
    return count_head, count_helmet

def calculate_distribution_head_and_helmet(xml_folder_path):
    xml_files = os.listdir(xml_folder_path)
    xml_files = [xml_file for xml_file in xml_files if xml_file.endswith('.xml')]
    count_head = 0
    count_helmet = 0
    for xml_file in xml_files:
        file_path = os.path.join(xml_folder_path, xml_file)
        count_head_temp, count_helmet_temp = count_tags_head_and_helmet(file_path)
        count_head += count_head_temp
        count_helmet += count_helmet_temp
    return count_head, count_helmet

def get_head_file(xml_folder_path):
    xml_files = os.listdir(xml_folder_path)
    xml_files = [xml_file for xml_file in xml_files if xml_file.endswith('.xml')]
    head_files = []
    for xml_file in xml_files:
        file_path = os.path.join(xml_folder_path, xml_file)
        count_head_temp, count_helmet_temp = count_tags_head_and_helmet(file_path)
        if count_head_temp > 0 and count_helmet_temp == 0:
            head_files.append(xml_file)
    return head_files

def get_helmet_file(xml_folder_path):
    xml_files = os.listdir(xml_folder_path)
    xml_files = [xml_file for xml_file in xml_files if xml_file.endswith('.xml')]
    helmet_files = []
    for xml_file in xml_files:
        file_path = os.path.join(xml_folder_path, xml_file)
        count_head_temp, count_helmet_temp = count_tags_head_and_helmet(file_path)
        if count_helmet_temp > 0 and count_head_temp == 0:
            helmet_files.append(xml_file)
    return helmet_files


