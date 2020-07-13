from LazierGreedy import *
from pickle import dump,load
import sys
from sys import argv
from datetime import datetime
from os import mkdir
from os.path import exists
if not exists("SimBA"):
    mkdir("SimBA")
path="SimBA/"+str(datetime.now())+"/"
mkdir(path)
sys.stdout=open(path+"log.txt","w")
Data={}
succ=0
tot=0
if argv[1]=="undef":
    from madryCifarUndefWrapper import *
    target_set=load(open("indices.pkl","rb"))
else:
    from madryCifarWrapper import *
    target_set=load(open("def_indices.pkl","rb"))
for j in target_set:
    tot+=1
    print("Starting attack on image", tot, "with index",j)
    ret=SimBA(x_test[j:j+1],mymodel,y_test[j],8/255,max_calls=20000, x_ent=True)
    dump(ret[1].reshape(1,32,32,3),open(path+"image_"+str(j)+".pkl","wb"))
    Data[j]=(ret[0],ret[2])
    if ret[0]:
        succ+=1
        print("Attack Succeeded with",ret[2],"queries, success rate is",100*succ/tot)
    else:
        print("Attack Failed using",ret[2],"queries, success rate is",100*succ/tot)
        dump(Data,open(path+"data.pkl","wb"))
