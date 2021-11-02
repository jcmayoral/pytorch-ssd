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
        image = self._read_image(image_id)
        boxes, labels, is_difficult = self._get_annotation(image_id, image.shape)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
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
        print ("INDEX",index)
        image_id = self.ids[index]
        print (image_id, "Annotation")
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        print(image_sets_file)
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id, image_size):
        annotation_file = image_id.replace('jpg','txt')
        #objects = ET.parse(annotation_file).findall("object")
        boxes = []                #x->width y->height
        labels = []
        is_difficult = []
        height, width, channels = image_size

        with open(annotation_file,'r') as objects:
            for object in objects:
                objectr = object.split()
                class_name = int(objectr[0])#object.find('name').text.lower().strip()
                #bbox = object.find('bndbox')
                cx1, cy1, cwidth, cheight  =  [float(i) for i in objectr[1:]]
                x1 = int(width * (cx1 - cwidth/2))
                y1 = int(height * (cy1 - cheight/2))
                x2 = int(width * (cx1 + cwidth/2))
                y2 = int(height * (cy1 - cheight/2))

                if (x1 > width) or (y1>height):
                    print ("objectr ", objectr)
                    print (annotation_file, objectr)
                    print (x2, width, cwidth)
                    print (y2, height, cheight)
                    print ("AAAAAAAAAA")

                if (x2 > width) or (y2>height):
                    print ("objectr ", objectr)
                    print (annotation_file, objectr)
                    print (x2, width, cwidth)
                    print (y2, height, cheight)
                    print ("BBBBBBBBBBb")

                #x->width y->height xmin,ymin
                #FOLLOWING VOC as the
                x1 = int(np.rint((cx1 - cwidth/2)*width))
                y1 = int(np.rint((cy1 - cheight/2)*height))
                x2 = int(np.rint((cx1 + cwidth/2)*width))
                y2 = int(np.rint((cy1 + cheight/2)*height))
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict[class_name])
                is_difficult_str = None #object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_file):
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
