from LazierGreedy import *
from pickle import dump,load
from sys import argv
import sys
if sys.argv[1]=="undef":
    from madryCifarUndefWrapper import *
    target_set=load(open("indices.pkl","rb"))
else:
    from madryCifarWrapper import *
    target_set=load(open("def_indices.pkl","rb"))
#the following need to be set appropriately to point to the directory containing attacked images and the datafile
dir="DSBatched/"
dic=load(open("data.pkl","rb"))
from datetime import datetime
from os import mkdir
from os.path import exists
if not exists("DSRefine"):
    mkdir("DSRefine")
path="DSRefine/"+str(datetime.now())+"/"
mkdir(path)
sys.stdout=open(path+"log.txt","w")
loss=False
grs=4
if len(argv)>2:
    loss=argv[2]=="xent"
if len(argv)>3:
    grs=int(argv[3])
Data={}
succ=0
tot=0
for j in target_set:
    tot+=1
    print("Starting attack on image", tot, "with index",j)
    ret=DSRefFromImage(x_test,j,dir,dic,mymodel,y_test[j],8/255,max_calls=20000, batch_size=64,x_ent=loss,gr_init=grs)
    dump(ret[1].reshape(1,32,32,3),open(path+"image_"+str(j)+".pkl","wb"))
    Data[j]=ret[0],ret[2],ret[3]
    if ret[0]:
        succ+=1
        print("Attack Succeeded with",ret[2],"queries, success rate is",100*succ/tot)
    else:
        print("Attack Failed using",ret[2],"queries, success rate is",100*succ/tot)
    dump(Data,open(path+"data.pkl","wb"))
