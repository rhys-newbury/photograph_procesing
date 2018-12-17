# Create Dataset Of Cropped Faces

import cv2 as cv2
import random

base_path = '/home/rhys/catkin_ws/src/darknet/yolotrain_face/WIDER_train/'
# for each picture do 4 random crops.
# randomly from left to right

def cropImage(path,index):
    
    img = cv2.imread(base_path + 'images/' + path + '.jpg')

    h, w, _ = img.shape
    
    w2 = w
    #direction = 0
    direction = random.randint(0, 1)
    #cropAmount = 0.4
    cropAmount = random.random() / 2.5 + 0.1
  
    if direction == 0:
       
        x = int(w * cropAmount)
        w -= x  

    else:
        #crop from print('a')right
        x = 0
        w -= int(w * cropAmount)

    crop_img = img[0:h, x:x+w].copy()

    label = base_path + 'labels/' + path + '.txt'
    lines = adjustBoundingBoxes(label, cropAmount, direction, crop_img,w2,h, w)
    
    image_path = base_path + 'images2/' + path + str(index)
    cv2.imwrite(image_path + ".jpg", crop_img)
    

    path = base_path + 'labels2/' + path + str(index)
    x = open(path + ".txt", 'w')
    x.writelines('\n'.join(lines))
    x.close()


def adjustBoundingBoxes(file, cropAmount, direction, image, width, height, new_width):
    new_lines = []   

    for line in open(file):
        [class_val, x, y, w, h] = map(float, line.split(" "))

        x = (x*width)
        w = (w*width)

        y = (y*height)
        h = (h*height)
     
        trueCrop = (cropAmount * width)
        if direction == 0:


            x -= trueCrop

            if (x - w//2) <= 0:
                class_val = 1
                w += x
                

        else:
            if (x + w//2) > new_width:

                class_val = 1
                w -= (x - new_width)

        top_left = (max(0, (x-w//2)), (y-h//2))
        bottom_right = (min(new_width, (x+w//2)), (y+h//2))

        width_bb = abs(top_left[0] - bottom_right[0])
        height_bb = abs(bottom_right[1] - top_left[1])

        centre_x = (top_left[0] + bottom_right[0]) / 2
        centre_y = (bottom_right[1] + top_left[1]) / 2

        if centre_x < 0:
            continue
   
        new_lines.append(str(class_val) + ' %0.6f %f %f %f' % (centre_x / new_width, centre_y / height, width_bb / new_width, height_bb / height))

    return new_lines   

import glob

num = len(glob.glob(base_path + "images/*.jpg"))
j = 0
for file in glob.glob(base_path + "images/*.jpg"):
    

    for i in range(4):

        cropImage(file.split("/")[9][:-4], i)
    j += 1
    print(float(j) / num * 100)
