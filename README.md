# inference_tools
 Runs inference on Json bbox predictions from Tensorflow ModelZoo or YOLOv4.

## Explanation of files::

## json_absolute_converter.py:
Converts the json file outputted from the Darknet YOLOv4 implementation (https://github.com/AlexeyAB/darknet) to text file format. 1 text file per image with the same name, with absolute coordinates of the bounding box. (Class, confidence, left, top, height, width).

## load_model.ipynb
Loads a tensorflow model, and runs inference on images. The output is saved as a .json file on the same format as the one from AlexeyB's darknet implementation. This allows this output to be used in the same way as the yolo output.

## evaluate_model.ipynb
Takes the output from json_absolute_converter.py and finds the COCO detection metrics.
