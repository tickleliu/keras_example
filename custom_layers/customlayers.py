
from __future__ import print_function
import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.layers import merge
from keras.layers.core import Lambda
import tensorflow as tf

class bilinear_pooling(Layer):
    def __init__(self,**kwargs):

        super(bilinear_pooling, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        print(x.shape)
        x_new=tf.transpose(x,perm=[0,3,1,2])
        print(x_new.shape)
        x_new=tf.reshape(x_new,[-1,1024,49])
        print(x_new.shape)
        x_new_T=tf.transpose(x_new,perm=[0,2,1])
        print(x_new_T.shape)
        phi_I=tf.matmul(x_new,x_new_T)
        print(phi_I.shape)
        phi_I=tf.reshape(phi_I,[-1,1024*1024])
        print(phi_I.shape)
        phi_I=tf.divide(phi_I,49.0)
        print(phi_I.shape)
        phi_I_new = tf.sign(phi_I)*(tf.sqrt(tf.abs(phi_I)+1e-12))
        print(phi_I_new.shape)
        z_l2=tf.nn.l2_normalize(phi_I_new,dim=1)
        print(z_l2.shape)

        return z_l2

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0],1024*1024])

class bilinear_pooling_two(Layer):
    def __init__(self,**kwargs):

        super(bilinear_pooling_two, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        print(len(x))
        x_new,y_new=x
        x_new=tf.transpose(x_new,perm=[0,3,1,2])
        print(x_new.shape)
        x_new=tf.reshape(x_new,[-1,1024,49])
        print(x_new.shape)
        x_new_T=tf.transpose(x_new,perm=[0,2,1])
        print(x_new_T.shape)

        y_new=tf.transpose(y_new,perm=[0,3,1,2])
        print(y_new.shape)
        y_new=tf.reshape(y_new,[-1,1024,49])
        print(y_new.shape)


        phi_I=tf.matmul(y_new,x_new_T)
        print(phi_I.shape)
        phi_I=tf.reshape(phi_I,[-1,1024*1024])
        print(phi_I.shape)
        phi_I=tf.divide(phi_I,49.0)
        print(phi_I.shape)
        phi_I_new = tf.sign(phi_I)*(tf.sqrt(tf.abs(phi_I)+1e-12))
        print(phi_I_new.shape)
        z_l2=tf.nn.l2_normalize(phi_I_new,dim=1)
        print(z_l2.shape)

        return z_l2

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0],1024*1024])