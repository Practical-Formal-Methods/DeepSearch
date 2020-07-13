import tensorflow as tf
from model import Model
from pickle import dump, load
from sys import argv
from tensorflow import keras
import numpy as np
_, (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test=x_test/255.0
x_test=x_test.reshape(10000,32,32,3)
y_test=y_test.reshape(-1)
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
mymodel=CompatModel("model_defended/")
