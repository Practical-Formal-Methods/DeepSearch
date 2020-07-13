import numpy as np 
from itertools import product
import heapq
from pickle import load
class Image:
    def __init__(self, base, model, true_class, epsilon, group_size=1, group_axes=[1,2], u_bound=1,l_bound=0,start_mode=-1, logits=False, x_ent=False,verbose=True):
        preprocess=lambda x:x.reshape(base.shape)
        self.orig_shape=base.shape
        self.calls=0
        self.group_axes=group_axes
        self.verbose=verbose
        self.group_size=group_size
        def predict(x):
            self.calls+=1
            return model.predict(preprocess(x)).reshape(-1)
        self.predict=predict
        self.true_class=true_class
        self.upper=np.clip(base.reshape(-1)+epsilon,l_bound,u_bound)
        self.lower=np.clip(base.reshape(-1)-epsilon,l_bound,u_bound)
        self.status=np.ones_like(self.lower)
        if start_mode>0:
            self.image=self.upper.copy()
        elif start_mode<0:
            self.image=self.lower.copy()
            self.status=self.status*-1
        else:
            self.image=base.reshape(-1)
            self.status=self.status*0
        self.gains=np.zeros_like(self.lower)
        self.gains_to=np.zeros_like(self.lower)
        def loss(image):
            res=self.predict(image)
            if np.argmax(res)!=true_class:
                return -50000
            if logits:
                if x_ent:
                    return res[true_class]-np.log(np.sum(np.exp(res)))
                else:
                    rest=np.ones_like(res)
                    rest[true_class]=0
                    return res.true_class-np.max(res[rest>0])
            else:
                if x_ent:
                    return np.log(res[true_class])
                else:
                    rest=np.ones_like(res)
                    rest[true_class]=0
                    return np.log(res[true_class])-np.log(np.max(res[rest>0]))
        self.loss_fn=loss
        self.loss=loss(self.image)
        self.stale=np.zeros_like(self.gains)
        self.rmap=preprocess(np.arange(len(self.image)))
    def get_indices(self,source):
        j=[]
        for i in reversed(self.orig_shape):
            j.append(source % i)
            source=source//i
        j=list(reversed(j))
        for i in self.group_axes:
            j[i]=j[i]-(j[i] % self.group_size)
        indices=[]
        for i in range(len(self.orig_shape)):
            if i in self.group_axes:
                indices.append(list(range(j[i],j[i]+self.group_size)))
            else:
                indices.append([j[i]])
        ret=[]
        for k in product(*indices):
            ret.append(self.rmap[k])
        return ret
    def get_pivots(self,direction):
        indices=[]
        ret=[]
        for i in range(len(self.orig_shape)):
            indices.append(list(range(0,self.orig_shape[i],self.group_size if i in self.group_axes else 1)))
        for k in product(*indices):
            if direction==0 or self.status[self.rmap[k]]==direction:
                ret.append(self.rmap[k])
        return ret
    def gain(self, index, force=False, no_update=False, direction=0):
        if direction==0:
            if self.status[index]>0:
                direction=-1
            else:
                direction=1
        if self.gains_to[index]==direction and (not force):
            return self.gains[index]
        pert=self.image.copy()
        pert[self.get_indices(index)]=self.lower[self.get_indices(index)] if direction<0 else self.upper[self.get_indices(index)]
        res=self.loss_fn(pert)
        self.gains_to[index]=direction
        self.gains[index]=res-self.loss
        self.stale[index]=0
        return res-self.loss
    
    def push(self, index, loss_diff, direction=0):
        if direction==0:
            if self.status[index]>0:
                direction=-1
            else:
                direction=1
        self.image[self.get_indices(index)]=self.lower[self.get_indices(index)] if direction<0 else self.upper[self.get_indices(index)]
        self.status[self.get_indices(index)]=direction
        self.stale+=1
        self.loss+=loss_diff
        if self.verbose:
            print("Pushing group of",self.group_size,"beginning at",index,"to",["lower bound,","","upper bound,"][direction+1],"Current loss is",self.loss,"and",self.calls,"calls have been made to the model.")
    def reset(self):
        self.stale=self.stale*0
        self.gains_to=self.gains_to*0
        self.loss=self.loss_fn(self.image)
        if self.verbose:
            print("Purging gains cache")
    
    def sample_indices(self, count, direction=0):
        lst=self.get_pivots(direction)
        try:
            return np.random.choice(lst,count,False)
        except ValueError:
            return lst
    
def two_way_fuzz(image,model,true_class,epsilon,samples_per_step,reset_every=-1,max_calls=-1,max_iters=-1):
    iters=0
    target=Image(image,model,true_class,epsilon)
    print("Initial loss is",target.loss)
    cont=lambda:(max_iters<0 or iters<max_iters) and (max_calls<0 or target.calls<max_calls) and target.loss>0
    while(cont()):
        iters+=1
        indices=target.sample_indices(samples_per_step)
        ls=sorted([(target.gain(i),i) for i in indices])
        if len(ls)<2:
            if ls[0][0]<0:
                target.push(ls[0][1],ls[0][0])
        else:
            sec_best=ls[1]
            best=target.gain(ls[0][1],True),ls[0][1]
            if best[0]<0:
                target.push(best[1],best[0])
        if reset_every>0 and iters % reset_every==0:
            target.reset()
    if target.loss>0:
        return False,image,target.calls
    return True,target.image,target.calls
def two_way_hier_fuzz(image,model,true_class,epsilon,samples_per_step,max_calls,max_faults=10):
    iters=0
    faults=0
    target=Image(image,model,true_class,epsilon,group_size=4)
    print("Initial loss is",target.loss)
    divs=[max_calls//20,(max_calls*5)//20,max_calls]
    jump=0
    cont=lambda:(max_calls<0 or target.calls<max_calls) and target.loss>0
    while(cont()):
        if target.calls>divs[jump]:
            jump=jump+1
            target.group_size=target.group_size//2
            target.reset()
        iters+=1
        indices=target.sample_indices(samples_per_step)
        ls=sorted([(target.gain(i),i) for i in indices])
        if len(ls)<2:
            if ls[0][0]<0:
                target.push(ls[0][1],ls[0][0])
                faults=0
            else:
                faults+=1
        else:
            sec_best=ls[1]
            best=target.gain(ls[0][1],True),ls[0][1]
            if best[0]<0:
                target.push(best[1],best[0])
                faults=0
            else:
                faults+=1
        if faults>=max_faults:
            target.reset()
            faults=0
    if target.loss>0:
        return False,image,target.calls
    return True,target.image,target.calls


def systematic_fuzz(image,model,true_class,epsilon,max_calls):
    target=Image(image,model,true_class,epsilon,group_size=4)
    print("Initial loss is",target.loss)
    divs=[max_calls//20,(max_calls*5)//20,max_calls]
    i=0
    while(target.loss>0 and target.calls<max_calls):
        pivots=target.get_pivots(0)
        gains=[(target.gain(p,True),p) for p in pivots]
        heapq.heapify(gains)
        while(target.calls<divs[min(i,2)] and target.loss>0):
            best=heapq.heappop(gains)
            sec_best=gains[0]
            new_best=(target.gain(best[1],True),best[1])
            if new_best<sec_best and new_best[0]<0:
                target.push(new_best[1],new_best[0])
                new_best=(target.gain(best[1],True),best[1])
                heapq.heappush(gains,new_best)
            elif new_best[0]<0:
                heapq.heappush(gains,new_best)
            else:
                break
        if i<2:
            target.group_size=target.group_size//2
            i=i+1
        target.reset()
    if target.loss>0:
        return False,image,target.calls
    return True,target.image,target.calls

def parsi_no_batch(image,model,true_class,epsilon,max_calls):
    print(true_class)
    target=Image(image,model,true_class,epsilon,group_size=4)
    print("Initial loss is",target.loss)
    while(target.loss>-10000 and target.calls<max_calls):
        print("Starting push to upper bound")
        pivots=target.get_pivots(-1)
        gains=[(target.gain(p,True),p) for p in pivots]
        heapq.heapify(gains)
        patience=1000
        while(target.loss>-10000 and target.calls<max_calls and patience>0 and len(gains)>0):
            best=heapq.heappop(gains)
            sec_best=(0,0) if len(gains)==0 else gains[0]
            new_best=(target.gain(best[1],True),best[1])
            if new_best<sec_best and new_best[0]<0:
                target.push(new_best[1],new_best[0],1)
                patience=1000
            elif new_best<sec_best and new_best[0]>0:
                break
            else:
                heapq.heappush(gains,new_best)
                patience=patience-1
        print("Starting push to lower bound")
        pivots=target.get_pivots(1)
        gains=[(target.gain(p,True),p) for p in pivots]
        heapq.heapify(gains)
        patience=1000
        while(target.loss>-10000 and target.calls<max_calls and patience>0 and len(gains)>0):
            best=heapq.heappop(gains)
            sec_best=(0,0) if len(gains)==0 else gains[0]
            new_best=(target.gain(best[1],True),best[1])
            if new_best<sec_best and new_best[0]<0:
                target.push(new_best[1],new_best[0],-1)
                patience=1000
            elif new_best<sec_best and new_best[0]>0:
                break
            else:
                heapq.heappush(gains,new_best)   
                patience=patience-1
        if target.group_size>1:
            target.group_size=target.group_size//2
        target.reset()
    if target.loss>-10000:
        return False,image,target.calls
    return True,target.image,target.calls

def DeepSearch(image,model,true_class,epsilon,max_calls):
    target=Image(image,model,true_class,epsilon,group_size=4,x_ent=True)
    print("Initial loss is",target.loss)
    while(target.loss>-10000 and target.calls<max_calls):
        selected=[]
        for x in target.get_pivots(0):
            if target.gain(x,True)<0:
                selected.append(x)
        for x in selected:
            target.push(x,0)
        target.loss=target.loss_fn(target.image)
        if target.group_size>1:
            target.group_size=target.group_size//2
        target.reset()
    if target.loss>-10000:
        return False,image,target.calls
    return True,target.image,target.calls


def DeepSearchBatched(image,model,true_class,epsilon,max_calls,batch_size=64,randomize=True,x_ent=False,gr_init=4):
    target=Image(image,model,true_class,epsilon,group_size=gr_init,x_ent=x_ent)
    print("Initial loss is",target.loss)
    while(target.loss>-10000 and target.calls<max_calls):
        selected=[]
        cur_batch=0
        all_pivots=target.get_pivots(0)
        if randomize:
            np.random.shuffle(all_pivots)
        for x in all_pivots:
            cur_batch+=1
            if target.gain(x,True)<0:
                selected.append(x)
            if cur_batch==batch_size:
                for x in selected:
                    target.push(x,0)
                target.loss=target.loss_fn(target.image)
                if target.loss<-10000:
                    return True,target.image,target.calls
                if target.calls>max_calls:
                    return False,image,target.calls
                cur_batch=0
                selected=[]
        for x in selected:
            target.push(x,0)
        target.loss=target.loss_fn(target.image)
        if target.group_size>1:
            target.group_size=target.group_size//2
        target.reset()
    if target.loss>-10000:
        return False,image,target.calls
    return True,target.image,target.calls

def parsi(image,model,true_class,epsilon,max_calls, batch_size=64, x_ent=False,gr_init=4):
    def make_batch(all):
        batch=[]
        for i in range(0,len(all),batch_size):
            batch.append(all[i:i+batch_size])
        return batch
    print(true_class)
    target=Image(image,model,true_class,epsilon,group_size=gr_init, x_ent=x_ent)
    print("Initial loss is",target.loss)
    while(target.loss>-10000 and target.calls<max_calls):
        pivots_all=target.get_pivots(0)
        np.random.shuffle(pivots_all)
        for pivots in make_batch(pivots_all):
            if target.calls>max_calls:
                return False,image,target.calls
            if target.loss<-10000:
                return True,target.image,target.calls
            gains=[(target.gain(p,True),p) for p in pivots if target.status[p]==-1]
            heapq.heapify(gains)
            patience=1000
            while(target.loss>-10000 and target.calls<max_calls and patience>0 and len(gains)>0):
                best=heapq.heappop(gains)
                sec_best=(0,0) if len(gains)==0 else gains[0]
                new_best=(target.gain(best[1],True),best[1])
                if new_best<sec_best and new_best[0]<0:
                    target.push(new_best[1],new_best[0],1)
                    patience=1000
                elif new_best<sec_best and new_best[0]>0:
                    break
                else:
                    heapq.heappush(gains,new_best)
                    patience=patience-1
            gains=[(target.gain(p,True),p) for p in pivots if target.status[p]==1]
            heapq.heapify(gains)
            patience=1000
            if target.calls>max_calls:
                return False,image,target.calls
            if target.loss<-10000:
                return True,target.image,target.calls
            while(target.loss>-10000 and target.calls<max_calls and patience>0 and len(gains)>0):
                best=heapq.heappop(gains)
                sec_best=(0,0) if len(gains)==0 else gains[0]
                new_best=(target.gain(best[1],True),best[1])
                if new_best<sec_best and new_best[0]<0:
                    target.push(new_best[1],new_best[0],-1)
                    patience=1000
                elif new_best<sec_best and new_best[0]>0:
                    break
                else:
                    heapq.heappush(gains,new_best)   
                    patience=patience-1
        if target.group_size>1:
            target.group_size=target.group_size//2
        target.reset()
    if target.loss>-10000:
        return False,image,target.calls
    return True,target.image,target.calls

def SimBA(image,model,true_class,epsilon,max_calls,x_ent=False):
    target=Image(image,model,true_class,epsilon,group_size=1,start_mode=0,x_ent=x_ent)
    print("Initial loss is",target.loss)
    ls=target.get_pivots(direction=0)
    np.random.shuffle(ls)
    for p in ls:
        up=target.gain(p,True,direction=1)
        if up<0:
            target.push(p,up,1)
            print("pushing",p,"up, loss is",target.loss)
        else:
            down=target.gain(p,True,direction=-1)
            if down<0:
                target.push(p,down,-1)
                print("pushing",p,"down, loss is",target.loss)
        if target.calls>max_calls:
            return False,image,target.calls
        if target.loss<-10000:
            return True,target.image,target.calls
    return False,image,target.calls

def DSRefBatched(image,model,true_class,epsilon,max_calls,batch_size=64,randomize=True,x_ent=False):
    h_calls=[]
    def Refine(image,adv,model,true,tol=0.0001):
        high=adv.reshape(image.shape)
        low=image.copy()
        calls_made=0
        gap=np.max(np.abs(high-low))
        while(gap>tol):
            mid=(high+low)/2
            pred=np.argmax(model.predict(mid))
            calls_made+=1
            if pred==true:
                low=mid
            else:
                high=mid
            gap=np.max(np.abs(high-low))
            print("Refining, gap is",gap)
        return high,calls_made
    success,new_image,calls=DeepSearchBatched(image,model,true_class,epsilon,max_calls,batch_size,randomize,x_ent)
    if success:
        dist=epsilon
        while True:
            prop_im,ex_calls=Refine(image,new_image,model,true_class)
            calls+=ex_calls
            new_dist=np.max(np.abs(prop_im-image))
            h_calls.append((calls,new_dist))
            if dist-new_dist>0.0002 and calls<max_calls:
                dist=new_dist
                success,new_image,ex_calls=DeepSearchBatched(image,model,true_class,dist,max_calls-calls,batch_size,randomize,x_ent)
                calls+=ex_calls
                if not success:
                    return True,prop_im,h_calls,new_dist
            else:
                return True,prop_im,h_calls,new_dist
    else:
        return False,image,[(calls,0)],epsilon
            
def DSRefFromImage(images,id,dir,dic,model,true_class,epsilon,max_calls,batch_size=64,randomize=True,x_ent=False, gr_init=4):
    h_calls=[]
    def Refine(image,adv,model,true,tol=0.0001):
        high=adv.reshape(image.shape)
        low=image.copy()
        calls_made=0
        gap=np.max(np.abs(high-low))
        while(gap>tol):
            mid=(high+low)/2
            pred=np.argmax(model.predict(mid))
            calls_made+=1
            if pred==true:
                low=mid
            else:
                high=mid
            gap=np.max(np.abs(high-low))
            print("Refining, gap is",gap)
        return high,calls_made
    #success,new_image,calls=DeepSearchBatched(image,model,true_class,epsilon,max_calls,batch_size,randomize,x_ent)
    image=images[id:id+1]
    success,calls=dic[id]
    new_image=load(open(dir+"image_"+str(id)+".pkl","rb")).reshape(image.shape)
    if success:
        dist=epsilon
        while True:
            prop_im,ex_calls=Refine(image,new_image,model,true_class)
            calls+=ex_calls
            new_dist=np.max(np.abs(prop_im-image))
            h_calls.append((calls,new_dist))
            if dist-new_dist>0.0002 and calls<max_calls:
                dist=new_dist
                success,new_image,ex_calls=DeepSearchBatched(image,model,true_class,dist,max_calls-calls,batch_size,randomize,x_ent,gr_init)
                calls+=ex_calls
                if not success:
                    return True,prop_im,h_calls,new_dist
            else:
                return True,prop_im,h_calls,new_dist
    else:
        return False,image,[(calls,0)],epsilon