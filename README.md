# Self-driving-car-CNN
Self driving car was implemented using opencv techniques and CNN model on a simulator.
## STEP 1

Udacity simulator was used to capture the image data with a csv file containing driving log.

Link to it : https://github.com/udacity/self-driving-car-sim

https://user-images.githubusercontent.com/71375214/129631267-a3a7ffc4-1166-4162-975b-8fc218f97dfc.mp4

## Step 2

Images with the steering angle was  considered to develop the model. Since the most of the time the car moves in the straight line the values would be 0 training with entire data will make the model to move in a straight line even though there is a curve. Therefore, balancing the data is a necessary step where some part of the data was cut off to bring the dataset to a normal distribution.

## Step 3

Preprocessing the images is performed before training the model using opencv techniques. The CNN regression model was then implemented to predict the steering angle of the car and would gain full access of the moving. (Here only the images from the front camera was used)

![self](https://user-images.githubusercontent.com/71375214/129692641-b0b2ced1-fec5-44ba-a1d2-3c5004e58961.png)


Once the model was built it was rigged up with the simulator to predict the steering angle according to images provide by the simulator.

https://user-images.githubusercontent.com/71375214/129631296-b3ce0b60-29d3-4df3-9ee9-b4ef93cb4b3f.mp4

