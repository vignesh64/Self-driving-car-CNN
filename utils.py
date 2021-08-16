import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random
### import data

def import_data(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    df = pd.read_csv(os.path.join('data', 'driving_log.csv'), names = columns)
    # print(df.head())
    # print(df['Centre'][0])
    #print(get_name(df.Centre[0]))
    df['Center'] = df['Center'].apply(get_name)
    # print(df.head())
    # print(df.shape)
    return df

def get_name(filepath):
    return filepath.split('\\')[-1]





### visualize data and balance

def balanceData(data, display=True):
    nbins = 41
    samplesperbin = 1225
    hist,bins = np.histogram(data['Steering'],nbins)
    #print(bins)
    if display:
        center = (bins[:-1]+bins[1:])*0.5
        print(center)
        plt.bar(center,hist,width = 0.06)
        plt.plot((-0.5,0.5),(samplesperbin,samplesperbin))
        plt.show()
    removelist=[]
    for i in range(nbins):
        bindatalist = []
        for j in range(len(data['Steering'])):
            if data['Steering'][j] >= bins[i] and data["Steering"][j] <= bins[i+1]:
                bindatalist.append(j) # it contains all the data in a particular bin
        bindatalist = shuffle(bindatalist)
        bindatalist = bindatalist[samplesperbin:]
        removelist.extend(bindatalist)
    print('length of data to be dropped',len(removelist))
    data.drop(data.index[removelist],inplace=True)
    print('remaining data',len(data))
    if display:
        hist, _ = np.histogram(data['Steering'], nbins)
        #print(center)
        plt.bar(center,hist,width = 0.06)
        plt.plot((-0.5,0.5),(samplesperbin,samplesperbin))
        plt.show()

    return data





###prepare data
def loaddata(path, data):
    images_path=[]
    steering=[]
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        #print(indexed_data)
        images_path.append(os.path.join(path,'IMG',indexed_data[0]))
        #print(os.path.join(path,'IMG',indexed_data[0]))
        steering.append(float(indexed_data[3]))
    images_path = np.asarray(images_path)
    steering = np.asarray(steering)
    return images_path,steering





### image augmentation
def image_augment(image,steering):
    image = mpimg.imread(image)
    #for pan
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.2, 0.2), 'y':(-0.2, 0.2)})
        image = pan.augment_image(image)
    #for zoom
    if np.random.rand() < 0.5:
        zoom  = iaa.Affine(scale=(1,1.5))
        image = zoom.augment_image(image)
    #for brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.3,1.4)) #0 is dark, 1 is normal, above 1 is brighter
        image = brightness.augment_image(image)
    #flip images
    if np.random.rand() < 0.5:
        image = cv2.flip(image,1)
        steering = -steering


    return image, steering

# imares, steer = image_augment('test.jpg',0)
# plt.imshow(imares)
# plt.show()
# print(steer)




### preprocess


def preprocess(image):
    #crop
    #image = mpimg.imread(image)
    image = image[60:135,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image  = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.resize(image,(200,66))
    image = image/255
    return image

# imares = preprocess(mpimg.imread('test.jpg'))
# plt.imshow(imares)
# plt.show()



##Batch generator
def batchgenerator(image_path,steering_path,batchsize,trainFlag):
    #print(image[0])
    while True:
        # print('hey')
        imgbatch = []
        steeringbatch = []
        for i in range(batchsize):
            index = random.randint(0,len(image_path)-1)
            # print('integer',index)
            if trainFlag:
                # print('index',index)
                # print('image',image_path[index])

                image,steering = image_augment(image_path[index],steering_path[index])


            else:
                image = mpimg.imread(image_path[index])
                steering = steering_path[index]
            image = preprocess(image)

            imgbatch.append(image)
            steeringbatch.append(steering)
        yield (np.asarray(imgbatch), np.asarray(steeringbatch))



### Model

def model():
    model = Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3), activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model

# model = load_model('carautomation.h5')
# image = preprocess(mpimg.imread('test.jpg'))
# image = np.asarray(image)
# image = image.reshape(-1,66,200,3)
# print(image.shape)
# print(model.predict(image))






