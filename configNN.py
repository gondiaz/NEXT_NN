from glob import glob
import os
from keras.optimizers import Adam
saved_weights=None
saved_model='/home/mmkekic/NEXT_NN/weights/weights_5mm_resnet_reco_cor/weights-29-0.3210.h5'
train = False
predict = True
predict_fname = '/home/mmkekic/NN/classification_DNN_output/data/data_10_bar/classification_6485_resnet_reco_cor_5.npz'
evaluate = False
train_data_dir = '/home/mmkekic/analysis/MC/DE_10bar_Tl/NN_datasets/reco_hits_dataset_cor/'
predict_data_dir='/home/mmkekic/analysis/data/6485/NN_dataset/reco_hits_dataset_cor/'
#train_data = glob(train_data_dir+'*.h5')
train_data=None
predict_data = glob(predict_data_dir+'*.h5')
nvalid=5000
lr=5e-5
decay = 1e-6
optimizer = Adam
nepochs=50
batch_size=24
tensorboard_dir= 'logs/logs_5mm_resnet_reco_cor'
checkpoint_dir = 'weights/weights_5mm_resnet_reco_cor'
test_data_dir  = None
predict_data_dir = None
threshold=0.95
voxel_dimension  = [5.,5.,5.]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
