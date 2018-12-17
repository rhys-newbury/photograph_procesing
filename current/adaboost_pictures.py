from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import math
import cv2

class MeanClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self):
        """
        Called when initializing the classifier
        """
    
    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.

        """
 
        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        boxes, h, w = analyse(x)
        return( True if self.sumThirdsWeighting(boxes, h, w) >= 0.7 else False )
     
    def predict(self, X, y=None):
        try:
            getattr(self, "treshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return([self._meaning(x) for x in X])

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X))) 

        
    #################################
    ## CALCULATE DISTANCE FROM INTERSECTIONS

    def getClosesetDistance(self, bounding_box_centre, intersections, normalisation):   
        return (math.sqrt(min(map(lambda (w, h): (bounding_box_centre[0] - w)**2 + (bounding_box_centre[1]-h)**2, intersections)))) / normalisation

    def calcBoxCenter(self, bounding_box):
        return (bounding_box[0] + bounding_box[2] / 2, bounding_box[1] + bounding_box[3] / 2)

    def sumThirdsWeighting(self, bounding_boxes, h, w):
        return reduce(lambda x, y: x + self.getClosesetDistance(self.calcBoxCenter(y), self.getThirdIntersections(h, w), (math.sqrt((h/3)**2 + (w/3)**2)) * len(bounding_boxes)),bounding_boxes, 0)
    #################################




def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def analyse(image_path):

    image = cv2.imread(image_path)
 
    Height, Width, _ = image.shape

    scale = 0.00392
    classes = "Faces"

    net = cv2.dnn.readNet('/home/rhys/catkin_ws/src/darknet/yolov3.weights',  '/home/rhys/catkin_ws/src/darknet/cfg/yolov3.cfg')

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