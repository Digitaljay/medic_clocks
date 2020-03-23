import glob
for i in range(1,501):
    if len(glob.glob(str(i)+'.jpg'))==0:
        print(i)
