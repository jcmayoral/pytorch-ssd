import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os

def parse_data(datafile):
    mydict = dict()
    with open(datafile,'r') as f:
        for line in f:
            key, value = line.split("=")
            if key and value:
                mydict[key]=value
    return mydict



class YOLODataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for YOLO data.
        Args:
            root: the root of the YOLO dataset, the directory must contain a data file.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        dict_data = parse_data(root)
        if is_test:
            image_sets_file = dict_data['valid'].strip()
        else:
            image_sets_file = dict_data['train'].strip()


        #This are actually files names
        self.ids = YOLODataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_names = {'0', '1', '2', '3'}
        self.class_dict = {i: int(i) for i, class_name in enumerate(self.class_names)}
        print ("CLASS DICT " , self.class_dict)

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        print ("LABELS ", labels)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        print(image_sets_file)
        with open(image_sets_file) as f:
            for line in f:
                print (line)
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            print ("NOTE TO YOU... remember the hack to test if this works\n")
            class_name = 'S0'
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])
                print(self.class_dict[class_name],"\n")
                print ("B")
                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
