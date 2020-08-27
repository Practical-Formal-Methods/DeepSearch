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
from imgntWrapper import *
from LazierGreedy import *
from pickle import dump,load
import sys
from sys import argv
from datetime import datetime
from os import mkdir
from os.path import exists
if not exists("Parsimonious"):
    mkdir("Parsimonious")
path="Parsimonious/"+str(datetime.now())+"/"
mkdir(path)
sys.stdout=open(path+"log.txt","w")
loss=False
grs=32
if len(argv)>1:
    loss=argv[1]=="xent"
if len(argv)>2:
    grs=int(argv[2])
Data={}
succ=0
tot=0
for j in tqdm(range(1000)):
    tot+=1
    print("Starting attack on image", j, "with index",inds[j])
    ret=parsi(x_test[j:j+1],mymodel,y_test[j],8/255,max_calls=10000, batch_size=64,x_ent=loss,gr_init=grs)
    dump(ret[1].reshape(1,256,256,3),open(path+"image_"+str(j)+".pkl","wb"))
    Data[j]=(ret[0],ret[2])
    if ret[0]:
        succ+=1
        print("Attack Succeeded with",ret[2],"queries, success rate is",100*succ/tot)
    else:
        print("Attack Failed using",ret[2],"queries, success rate is",100*succ/tot)
    dump(Data,open(path+"data.pkl","wb"))
