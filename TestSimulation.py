import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2


sio = socketio.Server()
app = Flask(__name__)
maxSpeed = 15


#preprocess the images
def preprocess(image):
    image = image[60:135,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image  = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.resize(image,(200,66))
    image = image/255
    print(image.shape)
    return image


###import image from simulator

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preprocess(image)
    image = np.asarray(image)
    image = image.reshape(-1,66,200,3)
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed #setting the threshold
    print('{} {} {}'.format(steering, throttle, speed))
    sendControl(steering, throttle)


## connect to simulator

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0) #intially 0 and 0 is sent for steering and speed



## send controls to the simulator
def sendControl(steering, throttle):
    sio.emit('steer', data = {'steering_angle': steering.__str__(),
                              'throttle': throttle.__str__()
                              })






if __name__ == '__main__':
    model = load_model('carautomation1.h5')
    model.summary()
    app = socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app) #port no is 4567 for this simulator
