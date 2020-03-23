f=open("labels.txt","r")
l = [line.strip() for line in f]
f.close()
l[56]=0
l[57]=7
l[58]=3
l[61]=2
l[62]=0
l[144]=1
l[145]=0
l[146]=8
l[165]=0
l[166]=2
l[457]=1
l[464]=0
l[475]=8
l[497]=0
l[498]=2
l[499]=0
f=open("labels.txt","w")
for i in l:
    f.write(str(i)+"\n")
f.close()
