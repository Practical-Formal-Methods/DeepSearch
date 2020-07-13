from QLNES import *
from pickle import dump,load
import sys
from sys import argv
from datetime import datetime
from os import mkdir
from os.path import exists
if not exists("QLNES"):
    mkdir("QLNES")
path="QLNES/"+str(datetime.now())+"/"
mkdir(path)
sys.stdout=open(path+"log.txt","w")
Data={}
succ=0
tot=0
if argv[1]=="undef":
    from madrySVHNUndefWrapper import *
    target_set=load(open("indices.pkl","rb"))
else:
    from madrySVHNWrapper import *
    target_set=load(open("def_indices.pkl","rb"))
for j in range(0,len(target_set),10):
    tot+=10
    print("Starting attack on batch", j//10)
    corr=np.argmax(mymodel.predict(x_test[target_set[j:j+10]]),1)==y_test.reshape(-1)[target_set[j:j+10]]
    ret=attack(mymodel,x_test[target_set[j:j+10]],100,0.001,0.001,0.001,5,1,0.9,8/255,ongoing=corr,max_queries=20000)
    dump(ret[0].reshape(10,32,32,3),open(path+"image_"+str(j)+"_to_"+str(j+9)+".pkl","wb"))
    for k in range(10):
        Data[target_set[j+k]]=(ret[2][k],ret[1][k])
    succ+=sum(ret[2])
    print("Success rate is",100*succ/tot)
    dump(Data,open(path+"data.pkl","wb"))
