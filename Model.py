
#!/usr/bin/env python
import csv
import cv2
import numpy as np


from keras.layers import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


images=[]
measurements=[]



# The code was executed in cells in the jupyter notebook



#Data Training of track 1 - 3 laps
lines=[]
with open ('TrainData/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))



for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='TrainData/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))



# Data Recording in the reverse direction 
lines=[]
with open ('TrainingData/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))


for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='TrainingData/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))



#Data Collection of track 2
lines=[]
with open ('DataTrack2/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))


for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='DataTrack2/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))



#Data collection of track 1
lines=[]
with open ('DataTrack1/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))
for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='DataTrack1/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))



#Data Collection of track 1 and 2 
lines=[]
with open ('Datatrain2/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))


for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='Datatrain2/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))


# Data collection at steep turns
lines=[]
with open ('DataFinal/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))


for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='DataFinal/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))


#Center Lane Driving Data Collection 
lines=[]
with open ('DataCenter/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))

for line in lines:
    source_path=line[0]
    filename=source_path.split('/')[-1]
    current_path='DataCenter/IMG/'+filename
    image= cv2.imread(current_path)
    image_RGB=image[...,::-1]
    images.append(image_RGB)
    measurement=float(line[3])
    measurements.append(measurement)

print(len(images))
print(len(measurements))


lines=[]
with open ('CenterData/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))

for line in lines:
    source_path=line[0]
    filename=source_path.split('/')[-1]
    current_path='CenterData/IMG/'+filename
    image= cv2.imread(current_path)
    image_RGB=image[...,::-1]
    images.append(image_RGB)
    measurement=float(line[3])
    measurements.append(measurement)

print(len(images))
print(len(measurements))


#Specific scenarios 
lines=[]
with open ('Final_Data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))

for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='Final_Data/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))


# Specific turn scenarios 

lines=[]
with open ('turnData/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))

for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='turnData/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))

lines=[]
with open ('Turn_final/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print(len(lines))

for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='Turn_final/IMG/'+filename
        image= cv2.imread(current_path)
        image_RGB=image[...,::-1]
        images.append(image_RGB)
    correction=0.25
    measurement=float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

print(len(images))
print(len(measurements))


X_train=np.array(images)
y_train=np.array(measurements)


# Data Visualization 
v=np.arange(-1,1.1,0.1)
plt.hist(measurements,bins=v)
plt.title('Steering Angles for Data Captured(All Data)')
plt.ylabel('Frequency')
plt.xlabel('Steering Values')
plt.grid()
plt.show()



# Model Architecture 


model=Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#model.add(Lambda(lambda x: (x/255.0)-0.5,input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0)-0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.8))
model.add(Dense(50))
model.add(Dropout(0.7))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
history_object=model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=6,verbose=1)
model.save('model.h59')

print(history_object.history.keys())

### the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()




# Final Model - model.h70
from keras.models import load_model
model = load_model('model.h69')

history_object=model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=1,verbose=1)
model.save('model.h70')

print(history_object.history.keys())

### training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# Model Visualization
from keras.utils.visualize_util import plot
from keras.models import load_model
model = load_model('model.h70')

graph=plot(model, to_file='model.png', show_shapes=True)










