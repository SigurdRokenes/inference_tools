import pandas as pd
import numpy as np
import sys
import os
import time
#from Tensorflow.workspace.json_yolo_converter import RESULT_PATH

DATASET_NAME = 'Porpoises'
IMAGE_PATH = 'C:/Users/sigur/OneDriveMS/FYS-3741-MASTER/data/data_yoloformat/test/images/'
###Faster RCNN

#RESULT_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/results/fasterrcnn_testresults/'
RESULT_PATH = "C:/Users/sigur/OneDriveMS/FYS-3741-MASTER/Tensorflow/workspace/inference_tools/"
RESULT_NAME = 'New_labels/Frcnn_dropout.json'
SAVE_PATH = "objMet_Labels/frcnn_yolo_pred/"
#SAVE_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/results/fasterrcnn_testresults/yolo_format/'

### YOLO
#RESULT_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/results/yolov4_testresults/annotations/'
#RESULT_NAME = 'coco_results.json'
#SAVE_PATH = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/results/yolov4_testresults/annotations/absolute_format/'

### EfficientDet_d1
#RESULT_PATH = "C:/Users/sigur/OneDriveMS/FYS-3741-MASTER/Tensorflow/workspace/inference_tools/"
#RESULT_NAME = 'New_labels/EffDetD1_nms.json'
#SAVE_PATH = RESULT_PATH+'objMet_Labels/effDet_yolo_pred/'


THRESHOLD = 0.1

SAVE = True

print(os.getcwd())
class JsonWriter:
    def __init__(self):
        try:
            self.df = pd.read_json(RESULT_PATH+RESULT_NAME)
            #self.df = pd.read_json(RESULT_NAME)
        except:
            print('File not found. Exiting...')
            exit()


    def find_char_location(self, original, character):
        return [index for index, char in enumerate(original) if char == character]


    def find_image_name(self, image):
        #print('Image:', image)
        path_index = self.find_char_location(image, '/')[-1] + 1
        #path = image[:path_index]
        image_name = image[path_index:]
        return image_name


    def populate_annotations(self, arr, threshold):
        #id, category_id, bbox = (0,0,0)

        detections = self.df[arr]
        #print(len(detections))

        valid_detections = detections
        #nr of valid detections
        #n = len(valid_detections)
        #coordinates
        bbox = [coords for coords in valid_detections['bbox']]
        confidence = [conf for conf in valid_detections['score']]
        class_id = [cl_id-1 for cl_id in valid_detections['category_id']]

        return class_id, confidence, bbox


    def write_yolo_file(self, name, id, conf, bbox, save_path=SAVE_PATH):
        n = len(id)
        with open(save_path+name[:-4]+'.txt', 'w') as file:
            for i in range(n):
                left, top, width, height = bbox[i]

                file.write('{} {} {} {} {} {} \n'.format(id[i], round(conf[i], 4), round(left),
                                                         round(top), round(width), round(height)))
        time.sleep(0.01)


    def __call__(self):
        
        image_ids = np.unique(self.df['image_id'])
        #print(image_ids.shape)
        n_img = len(image_ids)
        print(len(image_ids))
        print('Processing image:')
        for index, image in enumerate(image_ids):
            sys.stdout.write('\r {}%'.format(round((index+1)/n_img * 100)))

            #image_name = self.find_image_name(image)
            #print(image_name)
            class_id, confidence, bbox = self.populate_annotations(arr=(self.df['image_id'] == image),
                                                                   threshold=THRESHOLD)
            if SAVE:
                self.write_yolo_file(self.df['image_id'][index], class_id, confidence, bbox)
                time.sleep(0.01)
            sys.stdout.flush()
        #self.populate_dictionary()


def main():
    writer = JsonWriter()
    writer()


if __name__ == "__main__":
    main()


