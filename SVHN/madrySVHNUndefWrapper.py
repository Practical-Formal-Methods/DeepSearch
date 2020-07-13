import tensorflow as tf
from model import Model
from pickle import dump, load
from sys import argv
import numpy as np
from scipy.io import loadmat
data=loadmat("test_32x32.mat")
x_test=data['X']
y_test=data['y']
x_test=x_test.transpose(3,0,1,2)
x_test=x_test/255.0
y_test=y_test.reshape(-1)%10
class CompatModel:
    def __init__(self,folder):
        self.sess=tf.compat.v1.Session()
        model_file = tf.train.latest_checkpoint(folder)
        model=Model("eval")
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)
        self.model=model
        self.calls=0
    def predict(self,images):
        self.calls+=images.shape[0]
        res=np.exp(self.sess.run(self.model.pre_softmax,feed_dict={self.model.x_input:images*255,self.model.y_input:[1]}))
        return res/np.sum(res,axis=1).reshape(-1,1)
mymodel=CompatModel("model_undefended/")
