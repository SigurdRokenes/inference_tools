import tensorflow as tf
import numpy as np
import os
import time
import matplotlib
from matplotlib import patches, text, patheffects
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
from src.image_tools import ImageTools
import sys

##### CONFIG

###Not used
THRESHOLD = 0.3




### \END CONFIG



def load_image_to_numpy_array(path):
    """
    Load a single image to numpy array
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(io.BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(im_height, im_width, 3).astype(np.uint8)

def apply_threshold(detections, threshold):
    """
    Remove predictions with a lower confidence than a defined threshold
    """
    #To remove or not to remove confidence scores
    #valid_confidence_scores = detections['detection_scores'][0].numpy()>=threshold
    valid_confidence_scores = detections['detection_scores'][0].numpy()
    nr_detections = np.count_nonzero(valid_confidence_scores)
    bboxes =  detections['detection_boxes'][0].numpy()[range(nr_detections)]
    classes = detections['detection_classes'][0].numpy()[range(nr_detections)]
    confidence_scores = detections['detection_scores'][0].numpy()[range(nr_detections)]

    return nr_detections, confidence_scores, bboxes, classes

def run_inference(image_np, model):
    """
    Runs inference on a single image (numpy format)
    """
    input_tensor = np.expand_dims(image_np, 0)
    detections = model(input_tensor)
    return apply_threshold(detections, THRESHOLD)



def main():
    ### Load model from file:
    start_time = time.time()
    tf.keras.backend.clear_session()
    #Loads model
    detect_fn = tf.saved_model.load(MODEL_PATH)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Loaded model in: {} seconds'.format(elapsed_time))


    #image_dir = 'C:/Users/Sigurd/OneDriveMS/FYS-3741-MASTER/data/data_yoloformat/test_images_for_programming/'
    image_dir = IMAGE_PATH
    image_names = os.listdir(image_dir)


    json_result = []
    start_time = time.time()
    print('Processing Images:')
    for i in range(len(image_names)):
        sys.stdout.write('\r Image: {}, Completion: {}%'.format(i+1, round((i+1) / len(image_names)*100)))
        #find path to current image
        image_path = os.path.join(image_dir, image_names[i])
        #load image to numpy array
        image_np = load_image_to_numpy_array(image_path)
        #run inference
        nr_detections, confidence_scores, bboxes, labels = run_inference(image_np, detect_fn)
        
        test_image = ImageTools(bbox = bboxes, image_tensor = image_np,
                                                label = labels, figsize=(10,10),
                                                outline = False)
        
        if WRITE_TO_FILE:
            for i in range(nr_detections):
                category_id = int(labels[i])
                bbox = test_image.convert_bb(test_image.bbox)[i]
                bbox = list(bbox.astype('float'))
                score = confidence_scores[i].astype('float')

                json_result.append({
                                    "image_id": image_path,
                                    "category_id": category_id,
                                    "bbox": bbox,
                                    "score": score
                                    })
        

        #Plotting
        if PLOT:
            test_image.plot_sample()
        
        sys.stdout.flush()

    end_time = time.time()
    print('\n','Process complete in {} seconds \n'.format(round(end_time - start_time, 1)))

    print('Writing to file...' )
    if WRITE_TO_FILE:
        import json
        #print(len(json_result))
        with open('result.json', 'w') as outfile:
            outfile.write('[')
            outfile.write('\n')
            for i in range(len(json_result)):
                json.dump(json_result[i], outfile, separators=(',', ':'))
                if i < len(json_result) - 1:
                    outfile.write(',\n')
                else:
                    outfile.write('\n')
            outfile.write(']')

    print('Writing to file complete.')


if __name__ == "__main__":
    main()