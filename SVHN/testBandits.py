# Copyright 2020 Max Planck Institute for Software Systems

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from Bandits import *
from pickle import dump,load
import sys
from sys import argv
from datetime import datetime
from os import mkdir
from os.path import exists
if not exists("Bandits"):
    mkdir("Bandits")
path="Bandits/"+str(datetime.now())+"/"
mkdir(path)
sys.stdout=open(path+"log.txt","w")
Data={}
succ=0
tot=0
tile=2
Batch_size=40
if argv[1]=="undef":
    from madrySVHNUndefWrapper import *
    target_set=load(open("indices.pkl","rb"))
else:
    from madrySVHNWrapper import *
    target_set=load(open("def_indices.pkl","rb"))
for j in range(0,len(target_set),Batch_size):
    tot+=Batch_size
    print("Starting attack on batch", j//Batch_size)
    corr=np.argmax(mymodel.predict(x_test[target_set[j:j+Batch_size]]),1)==y_test.reshape(-1)[target_set[j:j+Batch_size]]
    ret=attack(mymodel,x_test[target_set[j:j+Batch_size]],0.1,0.1,tile,0.001,0.0001,8/255,corr,20000)
    dump(ret[0].reshape(Batch_size,32,32,3),open(path+"image_"+str(j)+"_to_"+str(j+Batch_size-1)+".pkl","wb"))
    for k in range(Batch_size):
        Data[target_set[j+k]]=(ret[2][k],ret[1][k])
    succ+=sum(ret[2])
    print("Success rate is",100*succ/tot)
    dump(Data,open(path+"data.pkl","wb"))
