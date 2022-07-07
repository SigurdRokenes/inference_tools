from tracemalloc import start
import tensorflow as tf
import numpy as np
from matplotlib import patches, text, patheffects
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont



class ImageTools:
    def __init__(self, bbox = None, image_tensor = None, label = None, figsize = (1,1),
                 text_displacement = 50, outline = False):
        """
        param:
            
        """
        
        self.img = image_tensor
        self.label = label
        self.figsize = figsize
        self.disp = text_displacement
        self.bbox = bbox
        self.outline = outline

    def convert_bb(self, bbox):
        """
        Convert bounding box coordinates from (ymin, xmin, ymax, xmax) to absolute form (xmin, ymax, w, h)
        for plotting in matplotlib.
        """
        boxes = bbox.copy()
        #set xmin as the first coordinate
        boxes[:,0] = (bbox[:,1])*self.img.shape[1]
        #set ymax as the second coordinate
        boxes[:,1] = bbox[:,0]*self.img.shape[0]
        #set width as (xmax - xmin)
        boxes[:,2] = (bbox[:,3] - bbox[:,1]) * self.img.shape[1]
        #set height as (ymax - ymin)
        boxes[:,3] = (bbox[:,2] -  bbox[:,0]) * self.img.shape[0]
        if boxes.shape[0] == 1 : return boxes

        return np.squeeze(boxes)
    

    def draw_outline(self, obj):
        """
        Draw outline around matplotlib objects (rects, text, etc.)
        """
        obj.set_path_effects([patheffects.Stroke(linewidth=4,  foreground='black'),
                              patheffects.Normal()])


    def img_show(self, ax = None, fname=None):
        """
        Show image
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.xaxis.tick_top()
        ax.imshow(self.img)

        return ax
    

    def draw_box(self, bb, ax):
        """
        Draw bounding box on image
        """
        patch = ax.add_patch(patches.Rectangle((bb[0], bb[1]), bb[2], bb[3],
                                                fill=False, edgecolor='red', lw = 2))

        if self.outline:
            self.draw_outline(patch)
    

    def draw_text(self, bb, ax, txt, disp):
        """
        Draw text on image
        """
        text = ax.text(bb[0],(bb[1]-disp),txt,verticalalignment='top',
                        color='white',fontsize=10,weight='bold')
        if self.outline:
            self.draw_outline(text)
    

    def plot_sample(self, ax = None, fname=None):
        """
        Plots sample with bounding box and text
        """
        bb = self.convert_bb(self.bbox)
        ax = self.img_show(ax=ax, fname=fname)
        
        
        for i in range(len(self.bbox)):
            self.draw_box(bb[i], ax)
            self.draw_text(bb[i], ax, txt = str(self.label[i]), disp = self.img.shape[0] * 0.05)

        


def main():
    import os
    path = 'examples/test_im/'
    for im_path in (os.listdir(path)):
        print(im_path)

if __name__ == '__main__':
    pass
