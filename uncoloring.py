import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
end_with=526
start_with=501
for i in range(start_with,end_with+1):
    img=Image.open("one_num/"+str(i)+".jpg")
    w,h=40,70
    img = np.array(img)
    img = img.sum(axis=2)
    for x in range(40):
        for y in range(70):
            img[y][x]=0 if img[y][x]<=240 else 1
    plt.imshow(img,cmap=plt.cm.binary)
    # plt.show()
    # print(img)
    plt.savefig("black_white/"+str(i)+".jpg")
    print(i)
