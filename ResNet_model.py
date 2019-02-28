import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.models               import Model, load_model
from keras.layers               import Input, Dense, MaxPooling3D, AveragePooling3D, Conv3D, Activation, Dropout, ZeroPadding3D, Flatten, Add, BatchNormalization
from keras.optimizers           import SGD, Adam
from keras                      import callbacks
from keras.regularizers         import l2, l1
from keras.initializers         import RandomNormal
from keras.utils.layer_utils    import print_summary
from keras                      import regularizers
from keras.utils import Sequence
from scipy.ndimage.interpolation import rotate

def rnpa_bottleneck_layer_simple(input_tensor, nb_filters, filter_sz, stage, init='glorot_normal', reg=0.0, use_shortcuts=True):
    """ Bottleneck layer of ResNet - preforms 1 convolution-relu-activation; filter_sz convlution-relu-activation;  1 convolution-relu-activation
    Parameters:
    input_tensor - keras tensor
    nb_filters - int
    filter_sz - int
    stage -int
    init - initialization of weights
    reg - l2 regularization
    use_shortcut - bool
    """
    nb_in_filters, nb_bottleneck_filters = nb_filters
    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = 'plus' + str(stage)
    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage>1: # first activation is just after conv1
        x = BatchNormalization(epsilon=1e-4, axis=-1,momentum=0.9, center=False, scale=False, name=bn_name+'a')(input_tensor)
        x = Activation('relu', name=relu_name+'a')(x)
    else:
        x = input_tensor
    x = Conv3D(
            filters=nb_bottleneck_filters, 
            kernel_size=(1,1,1),
            kernel_initializer=init,
            kernel_regularizer=l2(reg),
            use_bias=False,
            name=conv_name+'a'
        )(x)
    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = BatchNormalization(epsilon=1e-4, axis=-1,momentum=0.9, center=False, scale=False, name=bn_name+'b')(x)
    x = Activation('relu', name=relu_name+'b')(x)
    x = Conv3D(
            filters=nb_bottleneck_filters, 
            kernel_size=(filter_sz,filter_sz,filter_sz),
            padding='same',
            kernel_initializer=init,
            kernel_regularizer=l2(reg),
            use_bias = False,
            name=conv_name+'b'
        )(x)
    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = BatchNormalization(epsilon=1e-4, axis=-1,momentum=0.9, center=False, scale=False, name=bn_name+'c')(x)
    x = Activation('relu', name=relu_name+'c')(x)
    x = Conv3D(
            filters=nb_in_filters, 
            kernel_size=(1,1,1),
            kernel_initializer=init, 
            kernel_regularizer=l2(reg),
            use_bias=False,
            name=conv_name+'c'
        )(x)
    # merge
    if use_shortcuts:
        x = Add(name=merge_name)([x, input_tensor])
    return x

def ResNet(inputs, nb_classes=2, layer1_params=(3,64,2), res_layer_params=(3,32,32),
        final_layer_params=None, init='glorot_normal', reg=1e-6, use_shortcuts=True):
    
    """
    Parameters
    ----------
    input_dim : tuple of (C, H, W)
    nb_classes: number of scores to produce from final affine layer (input to softmax)
    layer1_params: tuple of (filter size, num filters, stride for conv)
    res_layer_params: tuple of (filter size, num res layer filters, num res stages)
    final_layer_params: None or tuple of (filter size, num filters, stride for conv)
    init: type of weight initialization to use
    reg: L2 weight regularization (or weight decay)
    use_shortcuts: to evaluate difference between residual and non-residual network
    """
    sz_L1_filters, nb_L1_filters, stride_L1 = layer1_params
    sz_res_filters, nb_res_filters, nb_res_stages = res_layer_params
     
    img_input = inputs
    #######
    # if input shape is not (32,32,32) consider padding with 0s:
    # x= ZeroPadding3D(padding=(1, 1, 1), data_format=None)(img_input)

    x = Conv3D(
        filters=nb_L1_filters, 
        kernel_size=(sz_L1_filters,sz_L1_filters,sz_L1_filters),
        padding='same',
        strides=(stride_L1, stride_L1,stride_L1),
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name='conv0'
    )(img_input)
    
    x = BatchNormalization(epsilon=1e-4, axis=-1,momentum=0.9, center=False, scale=False, name='bn0')(x)
    x = Activation('relu', name='relu0')(x)
    
    for stage in range(1,nb_res_stages+1):
        x = rnpa_bottleneck_layer_simple(
            x,
            (nb_L1_filters, nb_L1_filters),
            sz_res_filters, 
            stage,
            init=init, 
            reg=reg, 
            use_shortcuts=use_shortcuts
        )
        if stage % int(nb_res_stages/2)==0:
            nb_L1_filters=2*nb_L1_filters
            x = Conv3D(
                filters=nb_L1_filters, 
                kernel_size=(sz_L1_filters,sz_L1_filters,sz_L1_filters),
                padding='same',
                strides=(stride_L1, stride_L1,stride_L1),
                kernel_initializer=init,
                kernel_regularizer=l2(reg),
                use_bias=False,
                name='conv'+str(stage)
            )(x)
            
            x = BatchNormalization(epsilon=1e-4, axis=-1,momentum=0.9, center=False, scale=False, name='bnF'+str(stage))(x)
            x = Activation('relu', name='reluF'+str(stage))(x)
            
    sz_pool_fin=4
    x = MaxPooling3D((sz_pool_fin,sz_pool_fin,sz_pool_fin), name='ave_pool')(x)    
    x = Flatten(name='flat')(x)
    x = Dropout(.5, name='dropout_1')(x)
    x = Dense(32, activation='relu', name='FC_1', kernel_initializer='glorot_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.5, name='dropout_2')(x)
    x = Dense(units=1, activation='sigmoid', kernel_initializer='lecun_normal', name='final')(x)
    model=Model(img_input, x, name='rnpa')
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.00005), metrics=['accuracy'])

    return model

def datarot(vec, ang_rot, axes):
    vec=rotate(vec, ang_rot, axes=axes, reshape=False)
    return vec

class GenData(Sequence):
    def __init__(self, x_set, y_set, batch_size, weights_set=None, augmentation=False):
        self.x, self.y    = x_set, y_set
        self.batch_size   = batch_size
        self.augmentation = augmentation
        if weights_set is not None:
            self.weights = weights_set
        else:
            self.weights = np.ones(len(x_set))
    def __len__(self):
        return np.ceil(len(self.x) / float(self.batch_size)).astype('int')
    def __getitem__(self, idx):
        axes1=np.random.choice(3,2, replace=False)
        axes2=np.random.choice(3,2, replace=False)
        #those are random rotations
        angle1=15.*np.random.randint(12)
        angle2=15.*np.random.randint(12)
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_w=self.weights[idx * self.batch_size:(idx + 1) * self.batch_size]
        aug= np.random.randint(2)
        if (self.augmentation * aug):
            batch_x=batch_x.copy()
            for item in range(len(batch_x)):
                batch_x[item]=datarot(batch_x[item],angle1, axes1)
                batch_x[item]=datarot(batch_x[item],angle2, axes2)
        return np.array(batch_x), np.array(batch_y), np.array(batch_w)
    
# The data generator to be used as :
# data_gen=GenData(x_train,y_train,batch_size,augmentation=True)
# note that validation data is left outside manually

# The model is trained with:
# hist = model.fit_generator(data_gen, shuffle=True, epochs=20, initial_epoch=0, verbose=1, validation_data=(x_valid,y_valid), callbacks=lcallbacks)

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true,y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
def svs(y_true, y_pred):
    # any tensorflow metric
    specificity = 0.99
    value, update_op = tf.metrics.sensitivity_at_specificity(y_true,y_pred, specificity)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'svs' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

