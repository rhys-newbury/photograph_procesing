from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(50,50,3), activation='relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])