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
import numpy as np 
def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*np.exp(lr*g)
    neg = (1-real_x)*np.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1
def attack(model, images, queries_per_iter, delta, max_lr, min_lr, plateau_length, plateau_drop, momentum, epsilon, ongoing=None, max_queries=30000):
    if(queries_per_iter % 2 ==1):
        print("Queries per iteration must be even. Increasing by 1")
        queries_per_iter+=1
    imshape=images.shape
    original=images.copy()
    current_lr=np.ones(imshape[0])*max_lr
    dim=imshape[1]*imshape[2]*imshape[3]
    prior=np.zeros_like(images)
    truth=np.argmax(model.predict(images),axis=1)
    past_losses=[None for i in range(plateau_length)]
    if ongoing is None:
        ongoing=np.array([True for i in range(imshape[0])])
    def loss(x,ongoing):
        preds=model.predict(x)[np.arange(sum(ongoing)),truth[ongoing]]
        return -np.log(np.clip(preds,1e-12,1))
    queries=np.zeros(imshape[0])
    while(np.max(queries)<max_queries and np.any(ongoing)):
        old_prior=prior.copy()
        prior=np.zeros_like(images)
        losses=np.zeros(imshape[0])
        print("Query max:",np.max(queries),"images active:",sum(ongoing),"Linf distance:",np.max(np.abs(original[ongoing]-images[ongoing])))
        for i in range(queries_per_iter//2):
            u_vec=delta*np.random.normal(size=prior.shape)/(dim**0.5)
            uploss=loss((images+u_vec)[ongoing],ongoing)
            downloss=loss((images-u_vec)[ongoing],ongoing)
            lossderiv=(uploss-downloss)/(delta)
            losses[ongoing]+=lossderiv
            prior[ongoing]+=lossderiv.reshape(-1,1,1,1)*u_vec[ongoing]
        losses=losses/queries_per_iter
        past_losses.append(losses)
        past_losses=past_losses[1:]
        if past_losses[0] is not None:
            current_lr[past_losses[0]<past_losses[-1]]=current_lr[past_losses[0]<past_losses[-1]]/plateau_drop
            np.clip(current_lr,min_lr,max_lr)
        prior=old_prior*momentum+prior*(1-momentum)
        images[ongoing]=np.clip((images+np.sign(prior)*current_lr.reshape(-1,1,1,1))[ongoing],original[ongoing]-epsilon,original[ongoing]+epsilon)
        queries[ongoing]+=queries_per_iter
        ongoing=np.logical_and(ongoing,np.argmax(model.predict(images),1)==truth)
    return images, queries, np.logical_not(ongoing)
