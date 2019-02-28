# NEXT_NN
## Dependance:

keras/tensorflow

## Usage:

python main.py after setting parameters in config_NN.py

## Files:

* ResNet - function for constructing network and DataGenerator
* Net_utils - function to construct (calls functions from ResNet), train, test or evaluate network
* data_utils - functions to transform data into 3d voxels; in case of data labels are -1
* main - main function that uses parameters from config_NN
* config_NN - all configuration parameters 

## Parameters: 

* saved_weights=None or saved_weights file                                                                                                                       
* saved_model=None or saved_model h5                                     
* train = True if training, False if used trained model                                                                                                                           
* predict = True if predicting on data, False otherwise                                                                                                                        
* predict_fname = filename where to save the prediction (npz array with keys evtnum and y)                 
* evaluate = True if testing (on MC), False otherwise                                                                                                                        
* train_data  - list of h5 pandas dataframe (must be set if train = True)                                                                                                                                                                                                                    
* predict_data - list of h5 pandas dataframe   (must be set if predict = True) 
* evaluate_data - list of h5 pandas dataframe   (must be set if evaluate = True) 

*training parameters:
  * nvalid=5000 - number of data for validation - deducted from all training data                                                                                                                             
  * lr=5e-5     - initial learning rate                                                                                                                             
  * decay = 1e-6    - learning rate decay                                                                                                                        
  * optimizer = Adam  - optimizer (import from keras.optimizers)                                                                                                                       
  * nepochs=50  - number of epochs to train                                                                                                                           
  * batch_size=24   - batch size                                                                                                                        
  * tensorboard_dir= 'logs/logs_10mm_resnet_reco_cor'                                                                                       
  * checkpoint_dir = 'weights/weights_10mm_resnet_reco_cor'    
* data voxelize parameters:  
  * threshold=0.99   (minimal ratio of energy cointained in the window)                                                                                                                       
  * voxel_dimension  = [10.,10.,10.]                                                                                                                                                                                                       
  
