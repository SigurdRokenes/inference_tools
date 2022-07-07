import os
import json
import numpy as np
from PIL import Image

###CHANGE PATHS TO YOUR LIKING
RESULT_PATH = "Labels/"
RESULT_NAME = RESULT_PATH+"Annotations.json"
### SCHOOL COMPUTER
#image_path = "C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/Tensorflow/workspace/inference_tools/examples/test_im/"
#GT_path = "C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/Tensorflow/workspace/inference_tools/examples/test_labels/"



### LAPTOP
#image_path = "C:/Users/sigur/OneDriveMS/FYS-3741-MASTER/Tensorflow/workspace/inference_tools/examples/test_im/"
#GT_path = "C:/Users/sigur/OneDriveMS/FYS-3741-MASTER/Tensorflow/workspace/inference_tools/examples/test_labels/"
image_path = 'C:/Users/sigur/OneDriveMS/FYS-3741-MASTER/data/data_yoloformat/test/images/'
GT_path = 'C:/Users/sigur/OneDriveMS/FYS-3741-MASTER/data/data_yoloformat/test/annotations_1class/'
#GT_path = 'C:/Users/sigur/OneDrive - UiT Office 365/FYS-3741-MASTER/data/data_yoloformat/test_annotations_for_programming/'
### 


class JsonGTWriter:
    def __init__(self, image_path, GT_path):
        """
        image_path: (str): full path to folder of images
        GT_path:    (str): full path to folder of annotations in YOLO format (x,y,w,h) (relative)
        """
        self.GT_path = GT_path
        self.image_path = image_path
        self.image_names = os.listdir(image_path)
        self.GT_names = os.listdir(GT_path)
        self.jsonFormat = []
      

    def find_image_size(self, name):
        """
        Takes image name, and returns width / height of image
        """
        return Image.open(self.image_path+name).size

    def read_yolo_format(self):
        """
        """
        for name in self.GT_names:
            #load GT file to array
            file = np.loadtxt(self.GT_path+name)
            im_name = name[:-3]+'jpg'
            #find image width and height
            im_shape = np.array(self.find_image_size(im_name))
            #print(im_shape)

            #add extra dimension if only 1 bbox
            if file.ndim == 1:
                file = file[np.newaxis, ...]
            bbox = file[:, 1:]
            #print(bbox)
            bbox_abs = self.convert_to_absolute(bbox, im_shape = np.array(im_shape))
            #print(np.round(bbox_abs))
            cls_id = file[:, 0]

            #print(file.shape)
            for i in range(file.shape[0]):


                self.jsonFormat.append({
                                        'image_id':     im_name,
                                        "category_id":  int(cls_id[i]),
                                        "bbox":         np.round(bbox_abs[i]).astype(int).tolist()
                })
        #print(self.jsonFormat)
            #print(file)
            #for line in file.ndim:

            #try:
            #    assert(file.shape)
            #print(file.ndim)
            #for line in file:
                
            #with open(self.GT_path+name, 'r') as file:
            #    print(file)
    def convert_to_absolute(self, bbox, im_shape):
        """
        TODO: THIS X AND Y IS CENTERED. SHOULD NOT BE.
        ^ Done I think.
        input: list of coordinates in YOLO format
        """
        #old solution:
        """
        converted_bbox = bbox.copy()
        converted_bbox[:, 0] = (bbox[:, 0] - 0.5*bbox[:, 2])  * im_shape[0]
        converted_bbox[:, 1] = (bbox[:, 1] - 0.5*bbox[:, 3]) * im_shape[1]
        converted_bbox[:, 2] = (bbox[:, 0] + bbox[:, 2]) * im_shape[0]
        converted_bbox[:, 3] = (bbox[:, 1] + bbox[:, 3]) * im_shape[1]
        return converted_bbox
        """
        converted_bbox = bbox.copy()
        converted_bbox[:, 0] = (bbox[:, 0] - 0.5*bbox[:, 2])  * im_shape[0]
        converted_bbox[:, 1] = (bbox[:, 1] - 0.5*bbox[:, 3]) * im_shape[1]
        converted_bbox[:, 2] = (converted_bbox[:, 0] + bbox[:, 2]* im_shape[0]) 
        converted_bbox[:, 3] = (converted_bbox[:, 1] + bbox[:, 3]* im_shape[1]) 
        return converted_bbox


    
    def writeToFile(self):
        with open('{}'.format(RESULT_NAME), 'w') as outFile:
            outFile.write('[')
            outFile.write('\n')
            for i in range(len(self.jsonFormat)):
                json.dump(self.jsonFormat[i], outFile, separators=(',', ':'))
                if i < len(self.jsonFormat) - 1:
                    outFile.write(',\n')
                else:
                    outFile.write('\n')
            outFile.write(']')


def main():
    ex = JsonGTWriter(image_path, GT_path)
    ex.read_yolo_format()
    ex.writeToFile()

if __name__ == "__main__":
    main()

