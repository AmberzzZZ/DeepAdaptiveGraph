from keras.layers import Input, Reshape, Concatenate, Lambda, add
from keras.models import Model
from keras.engine import Layer
from keras.initializers import Constant
import keras.backend as K
import tensorflow as tf
from hrnet import HRNet
from dag import GCN_global, GCN_local


def PointRegressGCN(input_shape=(512,512,3), n_lmks=24, init_edges=None):

    inpt = Input(input_shape)
    init_lmks = Input((n_lmks, 2))

    print("--------building encoder--------")
    x = HRNet(input_shape)(inpt)    # (b,128,128,256)
    h, w, c = K.int_shape(x)[1:]
    encoder_out = Reshape((h*w,c))(x)     # (b,hw,c)

    print("----------building gcn----------")
    lmks = init_lmks              # tensor, updating lmk coords
    adj_affine = AdjMatrix((n_lmks,n_lmks), init_edges)(inpt)       # trainable edges, (b,N,N)
    adj_precise = AdjMatrix((n_lmks,n_lmks), init_edges)(inpt)
    # gcn global
    gcn_input = Lambda(interpolate, arguments={'grid_w': w, 'grid_y': h})([encoder_out, lmks])   # (b,N,c)
    delta_mat = Lambda(delta, arguments={'n_lmks': n_lmks})(lmks)
    theta = GCN_global(in_dim=256, out_dim=256, out_features=9, num_nodes=n_lmks)([gcn_input, adj_affine, delta_mat])    # (b,9)
    lmks = Lambda(prepective_transform, arguments={'n_lmks': n_lmks})([lmks, theta])    # (b,N,2)

    # gcn local
    for i in range(5):
        gcn_input = Lambda(interpolate, arguments={'grid_w': w, 'grid_y': h})([encoder_out, lmks])
        delta_mat = Lambda(delta, arguments={'n_lmks': n_lmks})(lmks)
        offset = GCN_local(in_dim=256, out_dim=256, out_features=2, num_nodes=n_lmks)([gcn_input, adj_precise, delta_mat])   # (b,N,2)
        lmks = add([lmks, offset])

    # model
    model = Model([inpt, init_lmks], lmks)
    return model


class AdjMatrix(Layer):
    def __init__(self, shape, init_edges=None, **kwargs):
        super(AdjMatrix, self).__init__(**kwargs)
        self.shape = shape
        self.init_edges = init_edges

        if self.init_edges is not None:
            assert init_edges.shape == shape, 'init value not match the shape'
            self.EDGE_INITIALIZER = Constant(init_edges)
        else:
            self.EDGE_INITIALIZER = 'ones'

    def build(self, input_shape):
        self.mat = self.add_weight(shape=self.shape,
                                   initializer=self.EDGE_INITIALIZER,
                                   name='adj')
        self.built = True

    def call(self, input):
        # repeat along b-axis
        batch_size = K.shape(input)[0]   # scalar
        mat = tf.TensorArray('float32', size=1, dynamic_size=True, element_shape=self.shape)   # (b,N,N)
        def loop_body(b, mat):
            mat = mat.write(b, self.mat, 'float32')
            return b+1, mat
        _, mat = K.control_flow_ops.while_loop(lambda b,*args: b<batch_size, loop_body, [0,mat])   # tensorArray
        mat = mat.stack()    # tensor
        return mat

    def compute_output_shape(self, input_shape):
        return (None,) + self.shape


def delta(lmks, n_lmks):
    # lmks: (b,N,2)
    distance = K.expand_dims(lmks, axis=2) - K.expand_dims(lmks, axis=1)   # (b,N,N,2)
    return Reshape((n_lmks, n_lmks*2))(distance)   # (b,N,2N)


def prepective_transform(args, n_lmks):
    # lmk_coords: (b,N,2)
    # theta: (b,9)
    lmk_coords, theta = args
    b = K.shape(theta)[0]
    ones = tf.ones((b,n_lmks,1))
    lmks = Concatenate()([lmk_coords, ones])   # (b,N,3)
    theta = Reshape((3,3))(theta)
    lmks = tf.matmul(lmks, theta)   # (b,N,3)
    lmks_x = lmks[...,0] / lmks[...,2]
    lmks_y = lmks[...,1] / lmks[...,2]
    lmks = K.stack([lmks_x,lmks_y], axis=-1)   # (b,N,2)
    return lmks


def interpolate(args, grid_w, grid_y):
    # cnn_feature: (b,hw,c)
    # lmk_coords: (b,N,2), normed xy
    cnn_feature, lmk_coords = args

    # tranverse sample
    batch_size = K.shape(cnn_feature)[0]
    N = K.int_shape(lmk_coords)[1]
    c = K.int_shape(cnn_feature)[-1]
    gcn_feature = tf.TensorArray('float32', size=1, dynamic_size=True, element_shape=(N,c))   # (b,N,c)

    def loop_body(b, gcn_feature):

        sample_feature = cnn_feature[b]   # (hw,c)
        sample_x = lmk_coords[b,:,:1]     # (N,1), grid-level xy
        sample_y = lmk_coords[b,:,1:]

        x0 = tf.floor(sample_x)     # (N,1)
        x1 = x0 + 1
        y0 = tf.floor(sample_y)
        y1 = y0 + 1

        w_00 = (x1 - sample_x) * (y1 - sample_y)    # (N,1), bilinear weight
        w_01 = (x1 - sample_x) * (sample_y - y0)
        w_10 = (sample_x - x0) * (y1 - sample_y)
        w_11 = (sample_x - x0) * (sample_y - y0)

        x0 = K.cast(K.clip(x0, 0, grid_w-1), 'int32')   # (N,1)
        x1 = K.cast(K.clip(x1, 0, grid_w-1), 'int32')
        y0 = K.cast(K.clip(x0, 0, grid_y-1), 'int32')
        y1 = K.cast(K.clip(y1, 0, grid_y-1), 'int32')

        node1_id = x0 + y0*grid_w    # (N,1), flattened id
        node2_id = x0 + y1*grid_w
        node3_id = x1 + y0*grid_w
        node4_id = x1 + y1*grid_w

        interp_feature = w_00 * tf.gather_nd(sample_feature, node1_id) + \
                         w_01 * tf.gather_nd(sample_feature, node2_id) + \
                         w_10 * tf.gather_nd(sample_feature, node3_id) + \
                         w_11 * tf.gather_nd(sample_feature, node4_id)       # (N,c)

        gcn_feature = gcn_feature.write(b, interp_feature, 'float32')
        return b+1, gcn_feature

    _, gcn_feature = K.control_flow_ops.while_loop(lambda b,*args: b<batch_size, loop_body, [0,gcn_feature])   # tensorArray
    gcn_feature = gcn_feature.stack()    # tensor
    return gcn_feature


if __name__ == '__main__':

    import numpy as np

    model = PointRegressGCN(input_shape=(512,512,3), n_lmks=24)
    model.summary()

    # x = Input((32,32))
    # y = AdjMatrix((24,24), init_edges=None)(x)
    # print(y)






