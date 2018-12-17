import cv2
import numpy as np
import math
import itertools
from functools import partial
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance as dist
import glob



def get_output_layers(net):
    
    layer_names = net.getLayerNames()

    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def analyse(image_path):

    image = cv2.imread(image_path)
 
    Height, Width, _ = image.shape

    scale = 0.00392
    classes = "Faces"

    net = cv2.dnn.readNet('/home/rhys/catkin_ws/src/darknet/backup/faces-tiny_300000.weights',  '/home/rhys/catkin_ws/src/darknet/cfg/faces-tiny.cfg')

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    # initialization
    confidences = []
    boxes = []
    conf_threshold = 0.5	
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
    
               center_x = int(detection[0] * Width)
               center_y = int(detection[1] * Height)
               w = int(detection[2] * Width)
               h = int(detection[3] * Height)
               x = center_x - w / 2
               y = center_y - h / 2
               confidences.append(float(confidence))

               boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

  
    return ([boxes[i[0]] for i in indices], Height, Width)

def calcBoxHeight(boxes):
    return map(lambda (x,y,w,h): h, boxes)


def frame_single(boxes, h, w):
    print(boxes)
    
    (center_x,center_y,box_w,box_h) = boxes[0]


    s = box_h * 1.5
    height_frame = 3 * s
    width_frame = 4 * s

    top_of_frame = center_y - 1.2 * s

    left_of_frame = center_x - 4/3 * s
    
    return (int(left_of_frame), int(top_of_frame), int(width_frame), int(height_frame))

def frame_group_pic(boxes, w_boxes, h_boxes, center_x, center_y, s):
    print('s: ' + str(s))
    print('width :' + str(4 * s))
    height_frame = 3 * s
    width_frame = 4 * s

    top_of_frame = center_y - 1.2 * s
    left_of_frame = center_x - 2*s

    return (int(left_of_frame), int(top_of_frame), int(width_frame), int(height_frame)) 




def frame_group(boxes, h, w):
    
    (x1, y1, x2, y2) = findExtremes(boxes)
    w_boxes, h_boxes, center_x, center_y = (x2-x1, y2-y1, (x2+x1)/2.0, (y1+y2)/2.0)


    if w_boxes > 1.6 * h_boxes:
        print('wide')
        return frame_group_pic(boxes, w_boxes, h_boxes, center_x, center_y,3/8.0 * w_boxes)
    else:
        return frame_group_pic(boxes, w_boxes, h_boxes, center_x, center_y, 1/2.0 * h_boxes)


def findExtremes(boxes):
    return reduce(lambda (x1,y1,x2,y2),(center_x,center_y,box_w,box_h): (min(x1, center_x - box_w / 2.0), min(center_y - box_h / 2.0, y1), max(x2, center_x + box_w / 2.0),  max(center_x + box_h / 2.0, y2)), boxes, [float('inf'),float('inf'),0,0])
        


    
def analyse_picture(file):

    boxes, h, w = analyse(file)

    print(len(boxes))

    img = cv2.imread(file)



    if len(boxes) == 1:
        (l, t, w, h) = frame_single(boxes, h, w)
   
      
    else:
        (l,t,w,h) = frame_group(boxes, h, w)


    crop = img.copy()
    print(l,t,w,h)
    crop = crop[t:t+h, l:l+w]

    print(file[2:-4] + "test.jpeg")

    cv2.imwrite(file[0:-4] + "test.jpeg", crop)


        



def main():
   
    for file in glob.glob('./*.jpg'):
        analyse_picture(file)

main()