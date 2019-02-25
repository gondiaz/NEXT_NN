from Resnet_model import *

def construct_model(saved_model=None, saved_weights=None, input_dimension=(32,32,32,1)):
    if load_model:
        model=load(saved_model)
    else:
        input_layer=Input(input_dimension)
        model = ResNet(input_layer)
        if saved_weights:
            model.load_weights(saved_weights)
    model.summary()
    return model

def train_model(model,nepoch,  x,y,nvalid=5000,data_gen=GenData,lr=5e-5,decay=1e-6, optimizer=Adam, tensorboard_dir=None, checkpoint_dir=None):
    adm=Adam(lr=lr,decay=decay)
    model.compile(loss='binary_crossentropy',optimizer=adm, metrics=['accuracy'])
    lcallbacks=[]
    if checkpoint_dir:
        file_lbl = "{epoch:02d}-{loss:.4f}"
        filepath=checkpoint_dir+'/weights-{}.h5'.format(file_lbl)
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        lcallbacks.append(checkpoint)
    if tensorboard_dir:
        tboard = callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=False)
        lcallbacks.append(tboard)
    
    X_,Y_=x[:,:,:,:,np.newaxis],y
    data_generator=data_gen(X_[:-nvalid],Y_[:-nvalid],24,rescale=False, augmentation=True)
    x_valid, y_valid = X_[-nvalid:],Y_[-nvalid:]
    hist = model.fit_generator(data_generator, shuffle=True, epochs=40, initial_epoch=0, verbose=1, validation_data=(x_valid,y_valid), callbacks=lcallbacks)


def evaluate_model(model, x_valid, y_valid):
    loss, met=model.evaluate(x_valid[:,:,:,:,np.newaxis],y_valid)
    return loss, met

def predict_model(model, x):
    y_pred=model.predict(x)
    return y_pred


