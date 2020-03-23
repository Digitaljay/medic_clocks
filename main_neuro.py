from keras import models
from keras import layers
import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical
import glob
end_with=500
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
train_data=train_data.reshape((len(train_data),70*40))
train_data=train_data.astype("float32")

print(train_data.shape)
print(train_data[0])

f=open("labels.txt","r")
train_labels = np.array([int(line.strip()) for line in f])
f.close()
train_labels=to_categorical(train_labels)
def build_model():

    network=models.Sequential()
    network.add(layers.Dense(32, activation="relu", input_shape=(70*40,)))
    network.add(layers.Dense(10,activation="softmax"))

    network.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=["acc"])

    return network


k=5
num_val_samples=len(train_data)//k
num_epochs=100
# all_scores=[]
all_acc_histories=[]

for i in range(k):
    print("Processing fold #",i)

    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_labels[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([train_labels[:i*num_val_samples],train_labels[(i+1)*num_val_samples:]],axis=0)

    model=build_model()
    history=model.fit(partial_train_data,
              partial_train_targets,
                      validation_data=(val_data,val_targets),
              batch_size=1,
              epochs=num_epochs,
              verbose=0)

    acc_history=history.history["val_acc"]
    all_acc_histories.append(acc_history)

average_mae_history=[np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]
print(average_mae_history)

