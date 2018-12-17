import glob
import cv2
itemCount = len(glob.glob("/home/rhys/catkin_ws/src/darknet/yolotrain_face/WIDER_train/labels2/*.txt"))

for index, file in enumerate(sorted(glob.glob("/home/rhys/catkin_ws/src/darknet/yolotrain_face/WIDER_train/images2/*.jpg"))):
    print(file)
    print(float(index) / itemCount * 100)
    img = None
    img = cv2.imread(file)
    
    height, width, _ = img.shape
    
    lbl = file.replace("jpg", "txt").replace("/images", "/labels")
    x = open(lbl)

    for line in x:
        [class_val, x, y, w, h] = map(float, line.split(" "))
        
        x = (x*width)
        w = (w*width)

        y = (y*height)
        h = (h*height)

        cv2.rectangle(img, (int(x-w//2), int(y-h//2)), (int(x+w//2), int(y+h//2)), (0,255,0) if int(class_val) == 0 else (255,0,0))
        print(line)

    cv2.imshow('img', img)
    cv2.waitKey(1000)
    
