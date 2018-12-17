import cv2
import numpy as np
import math
import itertools
from functools import partial
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance as dist



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



## Get the intersection points of the third lines.
def getThirdIntersections(h, w):

    h1 = h / 3
    h2 = 2 * h / 3

    w1 = w / 3
    w2 = 2 * w / 3

    return [(w1, h1), (w2, h2), (w2, h1), (w1, h2)]



#################################
## CALCULATE DISTANCE FROM INTERSECTIONS

def getClosesetDistance(bounding_box_centre, intersections, normalisation):   
    return (math.sqrt(min(map(lambda (w, h): (bounding_box_centre[0] - w)**2 + (bounding_box_centre[1]-h)**2, intersections)))) / normalisation

def calcBoxCenter(bounding_box):
    return (bounding_box[0] + bounding_box[2] / 2, bounding_box[1] + bounding_box[3] / 2)

def sumThirdsWeighting(bounding_boxes, h, w):
    return reduce(lambda x, y: x + getClosesetDistance(calcBoxCenter(y), getThirdIntersections(h, w), (math.sqrt((h/3)**2 + (w/3)**2)) * len(bounding_boxes)),bounding_boxes, 0)
#################################

#################################
## Calculate Distance from lines

def getClosesetDistanceToLine(bounding_box_centre, intersections, normalisation):   
    return math.sqrt(min(min(map(lambda (w, h): (bounding_box_centre[0] - w)**2, intersections)),min(map(lambda (w, h): (bounding_box_centre[1] - h)**2, intersections)))) / normalisation

def sumThirdsLinesWeighting(bounding_boxes, h, w):
    return reduce(lambda x, y: x + getClosesetDistanceToLine(calcBoxCenter(y), getThirdIntersections(h, w)[0:2], (min((h/3), (w/3))* len(bounding_boxes))),bounding_boxes, 0)
##################################

#################################
## Calculate distance from middle of picture

def getMiddlePunish(bounding_boxes, h, w):
    return reduce(lambda x,y : x + (1 / 20 * (abs(y[0]-w/2.0)) + 1 if abs(y[0] - w/2) < 20 else 0), map(calcBoxCenter, bounding_boxes), 0)
#################################

#################################

def extendDown(boxes, height):
    return [[x,y,w,(height-y)] for [x,y,w,h] in boxes]


#EXAMPLE OF findOverlap
# (0,_, 100, _), (80, _ 100, _) (0, _, 80, _), (80, _, 100, _)
# Assumes has  a sorted input
def findOverlap(bounding_boxes):
    
    readahead = iter(bounding_boxes)
    next(readahead)

    return [x[1][0] if x[0][0] + x[0][2] > x[1][0] else None for x in itertools.izip(bounding_boxes, readahead)] + [None]


def fixOverlap(bounding_boxes, changes):
    return [[x[0], x[1], changes[i], x[3]] if changes[i] != None else x for (i,x) in enumerate(bounding_boxes)]


def areaTakenUp(boxes, height, width):
    return reduce(lambda sum, (x, y, w, h): sum + w * h, fixOverlap(boxes, findOverlap(boxes)), 0) / (float(height * width))

def calcAreaWeight(boxes, height, width):
    return 9/4 * (areaTakenUp(boxes, height, width) - 1/3)**2
#################################


OCCUPANCY_WEIGHT = 1
MIDDLE_WEIGHT = 1
THIRDS_LINE_WEIGHT = 1
THIRDS_INT_WEIGHT = 1


def analyse_picture(image_path):

    boxes, h, w = analyse(image_path)
    print(len(boxes))


    if len(boxes) == 0:
        final_weight =  0
    else:

        funcs = [sumThirdsWeighting, sumThirdsLinesWeighting, getMiddlePunish, calcAreaWeight]
        factors = [THIRDS_INT_WEIGHT, THIRDS_LINE_WEIGHT, MIDDLE_WEIGHT, OCCUPANCY_WEIGHT]
        
        final_weight = reduce(lambda x,(i, y): x + factors[i] * y(boxes, h, w), enumerate(funcs), 0)

    print(image_path + " rating is " + str(final_weight))

import glob
def main():
   
    for file in glob.glob('/home/rhys/shaynedlima.github.io/image_crop/*.png'):
        analyse_picture(file)











main()







