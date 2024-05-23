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

xml_folder_path = r"D:\Code_school_nam3ki2\KhoaHocDuLieu\NhanDienMuBaoHiem\Tensorflow\workspace\images\Dataset_2\helm\helm\images\train"
count_head, count_helmet = calculate_distribution_head_and_helmet(xml_folder_path)
print(f"count_head: {count_head}, count_helmet: {count_helmet}")
