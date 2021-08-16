from utils import *
from sklearn.model_selection import train_test_split



### import data
path = 'data'
data = import_data(path)




### visual the data and balance

#print(data.Steering.value_counts())
data = balanceData(data,display=1)
print(len(data))


### prepare data

images, steering = loaddata(path,data)
#print(images[0], steering[0])


###split the data
x_train, x_val, y_train, y_val = train_test_split(images, steering, test_size=0.3, random_state=4)

#print('train and val shape:',x_train.shape, x_val.shape)




### Model

model = model()
model.summary()

##trainTraining the model


#history = model.fit(batchgenerator(x_train,y_train,200,1),epochs=10,steps_per_epoch=300,
#           validation_data=batchgenerator(x_val,y_val,200,0), validation_steps=200)
#
# model.save('carautomation.h5')
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['Training','val'])
# plt.ylim([0,0.5])
# plt.xlabel('epoch')
# plt.show()


##









