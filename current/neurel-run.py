import cv2
from keras.models import model_from_json

loaded_model = model_from_json("./model.json")
# load weights into new model
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

img = cv2.imread("./pics/0.jpg")

predictions = model.predict(img)