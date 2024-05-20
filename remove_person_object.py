from xml.etree import ElementTree as ET
import os
image_path_dir = r"D:\Code_school_nam3ki2\KhoaHocDuLieu\NhanDienMuBaoHiem\Tensorflow\workspace\images\test"

def read_xml_from_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return tree, root

def check_person_tag(root):
    person_tags = root.findall(".//object[name='person']")
    return len(person_tags) > 0

def remove_person_tags(tree, root, file_path):
    for person in root.findall(".//object[name='person']"):
        root.remove(person)
    tree.write(file_path)
list_xml = os.listdir(image_path_dir)
list_xml = [xml for xml in list_xml if xml.endswith(".xml")]

for xml in list_xml:
    xml_path = os.path.join(image_path_dir, xml)
    tree, root = read_xml_from_file(xml_path)
    if check_person_tag(root):
        remove_person_tags(tree, root, xml_path)
