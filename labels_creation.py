import os
f = open('labels.txt', 'a')
start_with=501
end_with=515
for i in range(start_with,end_with+1):
    os.startfile(str(i)+'.jpg')
    f.write(input()+"\n")
f.close()

