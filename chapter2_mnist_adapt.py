import numpy as np
from PIL import Image
import glob
from keras.utils.np_utils import to_categorical
end_with=len(glob.glob("one_num/*.jpg"))
start_with=1
# train_images=np.array([])
def b_w(i):
    img=Image.open("one_num/"+str(i)+".jpg")
    w,h=40,70
    img = np.array(img)
    img = img.sum(axis=2)
    for x in range(40):
        for y in range(70):
            img[y][x]=0 if img[y][x]<=240 else 1
    return img
train_data=np.array([b_w(i) for i in range(start_with,end_with+1)])
test_data=train_data[400:]
train_data=train_data[:400]

train_data=train_data.reshape((len(train_data),70*40))
test_data=test_data.reshape((len(test_data),70*40))

train_data=train_data.astype("float32")
test_data=test_data.astype("float32")

print(train_data.shape)
print(train_data[0])

f=open("labels.txt","r")
train_labels = [int(line.strip()) for line in f]
f.close()

test_labels=train_labels[400:]
train_labels=train_labels[:400]

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

# n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
# Y_train = np_utils.to_categorical(y_train, n_classes)
# Y_test = np_utils.to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", Y_train.shape)

from keras import models
from keras import layers
network=models.Sequential()
network.add(layers.Dense(32, activation="relu", input_shape=(40*70,)))
network.add(layers.Dense(10,activation="softmax"))
print("layers are ready")

network.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

network.fit(train_data,train_labels,epochs=5,batch_size=100)
# print("parametrs for compilation are chosen")
#
# train_images=train_images.reshape((60000,28*28))
# train_images=train_images.astype("float32")/255
# print("train_images prepared")
#
# test_images=test_images.reshape((10000,28*28))
# test_images=test_images.astype("float32")/255
# print("test_images prepared")
#
# from keras.utils import to_categorical
# train_labels=to_categorical(train_labels)
# test_labels=to_categorical(test_labels)
# print("labels are ready")
#
#
#
