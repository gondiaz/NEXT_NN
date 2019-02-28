from ResNet_model import *
from Net_utils import *
from data_utils import *
from configNN import *
import os
if __name__ == '__main__':
    model = construct_model(saved_model=saved_model, saved_weights=saved_weights)
    print ('model constructed \n')
    if train:
        if train_data is not None:
            print ('reading_data')
            x, y, evs = make_dataset (train_data, voxel_dimension, threshold=threshold, make_equal = True)
            print('training data size : ', len(x))
        else:
            print('must give training data')
            exit 
        print('training')
        train_model(model, nepochs, batch_size,x, y, nvalid=nvalid, data_gen = GenData, lr=lr, decay =decay, optimizer = optimizer, tensorboard_dir = tensorboard_dir, checkpoint_dir = checkpoint_dir)
    elif evaluate :
        if test_data is not None:
            print ('reading_data')
            x_test, y_test, evs_test = make_dataset (test_data, voxel_dimension, threshold=threshold, make_equal = False)
            print('test data size : ', len(x_test))
        loss, met = evaluate_model (model, x_test[:,:,:,:,np.newaxis], y_test)
        print('loss and metrics: ', loss, met)
        
    elif predict :
        if predict_data :
            x_pred, _, evs_pred = make_dataset (predict_data, voxel_dimension, threshold=threshold, make_equal=False)
            y_pred = predict_model (model, x_pred[:,:,:,:,np.newaxis])
            if predict_fname is not None:
                np.savez (predict_fname, y=y_pred, evtnum=evs_pred)
    exit 
