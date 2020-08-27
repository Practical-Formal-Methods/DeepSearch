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
def attack(model, images, exploration, delta, tile_size,prior_lr, image_lr, epsilon, ongoing=None, max_queries=30000):
    imshape=images.shape
    original=images.copy()
    prior=np.zeros((imshape[0],int(imshape[1]/tile_size),int(imshape[2]/tile_size),imshape[3]))
    upsample=lambda x:x.repeat(tile_size,1).repeat(tile_size,2)
    dim=imshape[1]*imshape[2]*imshape[3]/(tile_size*tile_size)
    truth=np.argmax(model.predict(images),axis=1)
    if ongoing is None:
        ongoing=np.array([True for i in range(imshape[0])])
    def prior_step(x, g):
        real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
        pos = real_x*np.exp(prior_lr*g)
        neg = (1-real_x)*np.exp(-prior_lr*g)
        new_x = pos/(pos+neg)
        return new_x*2-1
    def normalise(x):
        norms=np.sum(np.sum(np.sum((x*x),3),2),1)
        return x/norms.reshape(-1,1,1,1)
    def loss(x,ongoing):
        preds=model.predict(x)[np.arange(sum(ongoing)),truth[ongoing]]
        return -np.log(np.clip(preds,1e-12,1))
    queries=np.zeros(imshape[0])
    while(np.max(queries)<max_queries and np.any(ongoing)):
        print("Query max:",np.max(queries),"images active:",sum(ongoing),"Linf distance:",np.max(np.abs(original[ongoing]-images[ongoing])))
        u_vec=exploration*np.random.normal(0,1,prior.shape)/(dim**0.5)
        up=normalise(upsample(prior+u_vec))
        down=normalise(upsample(prior-u_vec))
        uploss=np.zeros(imshape[0])
        downloss=np.zeros(imshape[0])
        uploss[ongoing]=loss((images+delta*up)[ongoing],ongoing)
        downloss[ongoing]=loss((images+delta*down)[ongoing],ongoing)
        lossderiv=(uploss-downloss)/(delta*exploration)
        grad=lossderiv.reshape(-1,1,1,1)*u_vec
        prior=prior_step(prior,grad)
        images[ongoing]=np.clip((images+upsample(np.sign(prior))*image_lr)[ongoing],original[ongoing]-epsilon,original[ongoing]+epsilon)
        queries[ongoing]+=2
        ongoing=np.logical_and(ongoing,np.argmax(model.predict(images),1)==truth)
    return images,queries,np.logical_not(ongoing)
