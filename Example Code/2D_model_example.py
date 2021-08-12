# -*- coding: utf-8 -*-.

"""
@author: Andreas Wunsch, 2021
MIT Licencse
large parts from Sam Anderson (https://github.com/andersonsam/cnn_lstm_era)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle
from matplotlib import pyplot
from tensorflow.keras.callbacks import EarlyStopping
from random import seed
import os
from uncertainties import unumpy
from scipy import stats
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


#%% functions


def nse(y_obs, y_model):
    """Nash Sutcliffe Efficieny."""
    if not isinstance(y_obs, np.ndarray): #if tensor (or not array), convert to numpy array
      y_obs = np.array(y_obs)
    if not isinstance(y_model, np.ndarray):
      y_model = np.array(y_model)

    y_model = y_model.reshape((-1,1)) #make sure model and obs have same shape
    y_obs = y_obs.reshape((-1,1))

    nse = 1 - np.sum((y_model - y_obs)**2) / np.sum((y_obs - np.mean(y_obs))**2) #calculate NSE
    return nse


def nse_op(y_obs, y_model, ref):
    """
    Calculates modified Nash Sutcliffe Efficieny.
    Takes additional argument 'ref' which is the mean value of an alternaitve reference period than the test period of the model
    e.g. mean of the whole time series instead of the mean of the test set"""
    if not isinstance(y_obs, np.ndarray): #if tensor (or not array), convert to numpy array
      y_obs = np.array(y_obs)
    if not isinstance(y_model, np.ndarray):
      y_model = np.array(y_model)
    if not isinstance(ref, np.ndarray):
      ref = np.array(ref)

    y_model = y_model.reshape((-1,1)) #make sure model and obs have same shape
    y_obs = y_obs.reshape((-1,1))
    ref = ref.reshape((-1,1))

    nse = 1 - np.sum((y_model - y_obs)**2) / np.sum((y_obs - np.mean(ref))**2) #calculate NSE
    return nse

class MCDropout(tf.keras.layers.Dropout):
    #TODO add docstring
    def call(self, inputs):
        return super().call(inputs, training=True)


def build_model(learning_rate, Pnorm, steps_in, nchannels, n):
  #TODO add docstring
  model = Sequential()

  model.add(TimeDistributed(Conv2D(filters = 64, 
                                   kernel_size = (3,3), 
                                   activation='relu',
                                   data_format='channels_last', 
                                   padding='same'
                                   ), 
                            input_shape=(steps_in,)+np.shape(Pnorm[0])+(nchannels,)
                            )
            )
  
  model.add(TimeDistributed(Conv2D(filters = 64, 
                                   kernel_size = (3,3), 
                                   activation='relu',
                                   data_format='channels_last', 
                                   padding='same'
                                   )
                            )
            )
  
  model.add(TimeDistributed(MaxPooling2D(pool_size = 2,
                                         strides=(2,2))))
  
  model.add(TimeDistributed(Conv2D(filters = 128, 
                                   kernel_size = (3,3), 
                                   activation='relu',
                                   data_format='channels_last', 
                                   padding='same'
                                   )
                            )
            )
  
  model.add(TimeDistributed(Conv2D(filters = 128, 
                                   kernel_size = (3,3), 
                                   activation='relu',
                                   data_format='channels_last', 
                                   padding='same'
                                   )
                            )
            )

  model.add(TimeDistributed(MaxPooling2D(pool_size = 2,
                                         )))

  model.add(TimeDistributed(Flatten()))

  model.add(MCDropout(0.1))

  #1D-CNN
  model.add(Conv1D(filters=n,
                   kernel_size=3,
                   padding='same',
                   activation='relu'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(MaxPooling1D(padding='same'))
   
  #Flatten
  model.add(Flatten())
  
  #Dense
  model.add(Dense(1,activation = 'linear'))

  #compile
  model.compile(loss=tf.keras.losses.MSE,
                optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
  
  return model


def bayesOpt_function(n, steps_in, batchsize, inpT, inpTsin, inpSMLT, inpE, inpSF, inpSWVL1, inpSWVL2, inpSWVL3, inpSWVL4):
    #TODO add docstring
    n = 2**int(n)
    steps_in = 6*int(steps_in)
    batchsize = 2**int(batchsize)
    
    inpT = int(round(inpT))
    inpTsin = int(round(inpTsin))
    inpSMLT = int(round(inpSMLT))
    inpE = int(round(inpE))
    inpSF = int(round(inpSF))
    inpSWVL1 = int(round(inpSWVL1))
    inpSWVL2 = int(round(inpSWVL2))
    inpSWVL3 = int(round(inpSWVL3))
    inpSWVL4 = int(round(inpSWVL4))
    
    return bayesOpt_function_with_discrete_params(n, steps_in, batchsize, inpT, inpTsin, inpSMLT, inpE, inpSF, inpSWVL1, inpSWVL2, inpSWVL3, inpSWVL4)

def bayesOpt_function_with_discrete_params(n, steps_in, batch_size, inpT, inpTsin, inpSMLT, inpE, inpSF, inpSWVL1, inpSWVL2, inpSWVL3, inpSWVL4):
    
    #TODO add docstring
    
    learning_rate = 1e-3
    training_epochs = 100   
    earlystopping_patience = 10
    
# =============================================================================
#### construct train and test predictor/target tensors
# =============================================================================
    nchannels = 1 + inpT + inpTsin + inpSMLT + inpE + inpSF + inpSWVL1 + inpSWVL2 + inpSWVL3 + inpSWVL4
    
    y_train = np.squeeze([Qnorm[steps_in:trainInds[-1]+1,]]).T
    y_val = np.squeeze([Qnorm[valInds,] ]).T
    y_opt = np.squeeze([Qnorm[optInds,] ]).T
    
    y_train = y_train.astype(dtype = np.float16)
    y_val = y_val.astype(dtype = np.float16)
    y_opt = y_opt.astype(dtype = np.float16)
    
    x_intermediate = np.empty(np.shape(Pnorm) + (nchannels,),dtype='single')
    
    x_intermediate[:,:,:,0] = Pnorm
   
    idx = 1
    if inpT: 
        x_intermediate[:,:,:,idx] = Tnorm
        idx = idx+1
    if inpSMLT:
        x_intermediate[:,:,:,idx] = SMLTnorm
        idx = idx+1
    if inpTsin:
        x_intermediate[:,:,:,idx] = Tsinnorm
        idx = idx+1
    if inpE:
        x_intermediate[:,:,:,idx] = Enorm
        idx = idx+1
    if inpSF:
        x_intermediate[:,:,:,idx] = SFnorm
        idx = idx+1
    if inpSWVL1:
        x_intermediate[:,:,:,idx] = SWVL1norm
        idx = idx+1
    if inpSWVL2:
        x_intermediate[:,:,:,idx] = SWVL2norm
        idx = idx+1
    if inpSWVL3:
        x_intermediate[:,:,:,idx] = SWVL3norm
        idx = idx+1
    if inpSWVL4:
        x_intermediate[:,:,:,idx] = SWVL4norm
        idx = idx+1
    
    x_train = np.empty((Ntrain-steps_in, steps_in, ) + np.shape(Pnorm)[1:] + (nchannels,),dtype=np.float16)
    x_val = np.empty((Nval, steps_in,) + np.shape(Pnorm)[1:] + (nchannels,), dtype = np.float16)
    x_opt = np.empty((Nopt, steps_in,) + np.shape(Pnorm)[1:] + (nchannels,),dtype=np.float16)
    
    #training
    for ii in range(Ntrain-steps_in):
      x_train[ii] = x_intermediate[ii:ii+steps_in]
    # #validation
    for ii in range(Nval):
      x_val[ii] = x_intermediate[ii + Ntrain - steps_in : ii + Ntrain]
    # #optimizing ()
    for ii in range(Nopt):
      x_opt[ii] = x_intermediate[ii + Ntrain + Nval - steps_in : ii + Ntrain + Nval]
    
    # #convert predict/target arrays to tensors
    x_train = tf.convert_to_tensor(x_train)
    x_val = tf.convert_to_tensor(x_val)
    x_opt = tf.convert_to_tensor(x_opt)
    y_train = tf.convert_to_tensor(y_train)
    y_val = tf.convert_to_tensor(y_val)
    y_opt = tf.convert_to_tensor(y_opt)
    
    #create train/val/opt datasets for model
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(Ntrain).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(Nval).batch(batch_size)
    # opt_dataset = tf.data.Dataset.from_tensor_slices((x_opt, y_opt)).shuffle(Nopt).batch(batch_size)
    
# =============================================================================
#### training
# =============================================================================
    with tf.device("/gpu:2"):
        #define early stopping callback to use in all models
        callbacks = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              verbose=1, 
                              patience = earlystopping_patience,
                              restore_best_weights = True)

        inimax = 1
        sim = np.empty((Nopt,inimax))
        for ini in range(inimax):
            #generate/train model
            seed(ini+37657)
            tf.random.set_seed(ini+37657)
            
            model = build_model(learning_rate, Pnorm, steps_in, nchannels, n)
            # model.summary()
            history = model.fit(
                train_dataset,
                validation_data = val_dataset,
                epochs = training_epochs,
                verbose = 1, 
                callbacks = [callbacks]
                )
            
            # plot loss during training
            pyplot.figure(figsize=(10,4))
            pyplot.title('Loss')
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='val')
            pyplot.ylabel('Loss', size=12)
            pyplot.xlabel('Epochs',size=12)
            pyplot.legend()
            pyplot.show()
            
            sim1 = model.predict(x_opt, batch_size = batch_size, verbose = 0)
            sim[:,ini] = sim1.reshape(-1,)
    
    
    
        
    y_opt_sim = np.median(sim,axis = 1)
    pyplot.plot(y_opt,'k')
    pyplot.plot(y_opt_sim,alpha=0.7)
    pyplot.show()
    err = y_opt_sim-y_opt
    MSE = np.mean(err ** 2)

    return (-1)*MSE

def predict_distribution(X, model, n, batch_size):
#TODO add docstring
    preds = [model.predict(X, batch_size = batch_size) for _ in range(n)]
    return np.hstack(preds)

class newJSONLogger(JSONLogger) :
#TODO add docstring
      def __init__(self, path):
            self._path=None
            super(JSONLogger, self).__init__()
            self._path = path if path[-5:] == ".json" else path + ".json"


#%% define paths and load data


dir_data = './data_pickle' #where to save trained model outputs
dir_models = './Results' #where to save trained model outputs
dir_output = dir_models

os.chdir(dir_output)

# load data, which is already preprocessed and is a pickled dictionary with format:
#   'date': Datetimeindex (No_of_timesteps,)
#   'Variable': list (No_of_timesteps,)
#       each line of 'Variable' contains an array with dimensions (X_cells,Y_cells) (grid for each timestep)
    
# one pickle file for each variable
pickle_in = open(dir_data + '/' + 'TDict.pickle','rb')
tempDict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'TsinDict.pickle','rb')
tsinDict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'PDict.pickle','rb')
precDict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'SMLTDict.pickle','rb')
snowmeltDict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'EDict.pickle','rb')
EDict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'SFDict.pickle','rb')
SFDict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'SWVL1Dict.pickle','rb')
SWVL1Dict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'SWVL2Dict.pickle','rb')
SWVL2Dict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'SWVL3Dict.pickle','rb')
SWVL3Dict = pickle.load(pickle_in)

pickle_in = open(dir_data + '/' + 'SWVL4Dict.pickle','rb')
SWVL4Dict = pickle.load(pickle_in)

T = np.asarray(tempDict['T'])
Tsin = np.asarray(tsinDict['Tsin'])
SMLT = np.asarray(snowmeltDict['SMLT'])
P = np.asarray(precDict['P'])
E = np.asarray(EDict['E'])
SF = np.asarray(SFDict['SF'])
SWVL1 = np.asarray(SWVL1Dict['SWVL1'])
SWVL2 = np.asarray(SWVL2Dict['SWVL2'])
SWVL3 = np.asarray(SWVL3Dict['SWVL3'])
SWVL4 = np.asarray(SWVL4Dict['SWVL4'])

# pickle file for Q contains only an array ('Q' time series) and a datetimeindex ('date')
pickle_in = open(dir_data + '/' + 'QDict.pickle','rb')
QDict=pickle.load(pickle_in)
Q = np.asarray(QDict['Q']) 

t = QDict['date']


#%% split data


#years/indices of training, early stopping (validation), optimization and testing
#change years accordingly to you data
trainStartYear = 2012
trainFinYear = 2017

valStartYear = 2018
valFinYear = 2018

optStartYear = 2019
optFinYear = 2019

testStartYear = 2020
testFinYear = 2020

trainInds = np.squeeze(np.argwhere((t.year>=trainStartYear) & (t.year<=trainFinYear)))
valInds = np.squeeze(np.argwhere((t.year>=valStartYear) & (t.year<=valFinYear)))
optInds = np.squeeze(np.argwhere((t.year>=optStartYear) & (t.year<=optFinYear)))
testInds = np.squeeze(np.argwhere((t.year>=testStartYear) & (t.year<=testFinYear)))

refInds = np.squeeze(np.argwhere((t.year<testStartYear))) # for NSEop
                                 
Ntrain = len(trainInds)
Nval = len(valInds)
Nopt = len(optInds)
Ntest = len(testInds)

#scaling
scaler = StandardScaler()

Tnorm = scaler.fit_transform(T.reshape(-1, T.shape[-1])).reshape(T.shape)
SMLTnorm = scaler.fit_transform(SMLT.reshape(-1, SMLT.shape[-1])).reshape(SMLT.shape)
Pnorm = scaler.fit_transform(P.reshape(-1, P.shape[-1])).reshape(P.shape)
Tsinnorm = scaler.fit_transform(Tsin.reshape(-1, Tsin.shape[-1])).reshape(Tsin.shape)
Enorm = scaler.fit_transform(E.reshape(-1, E.shape[-1])).reshape(E.shape)
SFnorm = scaler.fit_transform(SF.reshape(-1, SF.shape[-1])).reshape(SF.shape)
SWVL1norm = scaler.fit_transform(SWVL1.reshape(-1, SWVL1.shape[-1])).reshape(SWVL1.shape)
SWVL2norm = scaler.fit_transform(SWVL2.reshape(-1, SWVL2.shape[-1])).reshape(SWVL2.shape)
SWVL3norm = scaler.fit_transform(SWVL3.reshape(-1, SWVL3.shape[-1])).reshape(SWVL3.shape)
SWVL4norm = scaler.fit_transform(SWVL4.reshape(-1, SWVL4.shape[-1])).reshape(SWVL4.shape)


Qscaler = StandardScaler()
Qscaler.fit(pd.DataFrame(Q))
Qnorm = Qscaler.transform(pd.DataFrame(Q))


#%% Bayesian Optimization:


# Bounded region of parameter space
pbounds = {'steps_in': (1,10*4),
           'n': (6,8),
           'batchsize': (6,9),
           'inpT': (0,1),
           'inpTsin': (0,1),
           'inpSMLT': (0,1), 
           'inpE': (0,1),
           'inpSF': (0,1),
           'inpSWVL1': (0,1),
           'inpSWVL2': (0,1),
           'inpSWVL3': (0,1),
           'inpSWVL4': (0,1)}

optsteps1 = 10 # random initial steps
optsteps2 = 30 # least no of steps
optsteps3 = 5 # how many steps no improvement
optsteps4 = 40 # max no of steps

optimizer = BayesianOptimization(
    f= bayesOpt_function, #Function that is optimized
    pbounds=pbounds, #Value ranges in which optimization is performed
    random_state=1, 
    verbose = 0 # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent, verbose = 2 prints everything
    )

# #load existing optimizer logs from previous runs
log_already_available = 0
if os.path.isfile("./logs.json"):
    load_logs(optimizer, logs=["./logs.json"]);
    print("\nExisting optimizer is already aware of {} points.".format(len(optimizer.space)))
    log_already_available = 1

# Save progress
logger = newJSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

if log_already_available == 0:
    optimizer.maximize(
            init_points=optsteps1, #steps of random exploration (random starting points before bayesopt(?))
            n_iter=0, # steps of bayesian optimization
            acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
            xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
            )

# optimize while improvement during last 10 steps
current_step = len(optimizer.res)
beststep = False
step = -1
while not beststep:
    step = step + 1
    beststep = optimizer.res[step] == optimizer.max #aktuell beste Iteration suchen

while current_step <= optsteps1: 
        current_step = len(optimizer.res)
        beststep = False
        step = -1
        while not beststep:
            step = step + 1
            beststep = optimizer.res[step] == optimizer.max
        print("\nbeststep {}, current step {}".format(step+1, current_step+1))
        print("new exploration step")
        optimizer.maximize(
            init_points=0, #steps of random exploration (random starting points before bayesopt(?))
            n_iter=1, # steps of bayesian optimization
            acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
            xi=0.1  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
            )
        
while current_step < optsteps2: 
        current_step = len(optimizer.res)
        beststep = False
        step = -1
        while not beststep:
            step = step + 1
            beststep = optimizer.res[step] == optimizer.max
        print("\nbeststep {}, current step {}".format(step+1, current_step+1))
        optimizer.maximize(
            init_points=0, #steps of random exploration (random starting points before bayesopt(?))
            n_iter=1, # steps of bayesian optimization
            acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
            xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
            )
        
while (step + optsteps3 > current_step and current_step < optsteps4): # Abbruch bei 50 Iterationen oder wenn seit 10 keine Verbesserung mehr eingetreten ist
        current_step = len(optimizer.res)
        beststep = False
        step = -1
        while not beststep:
            step = step + 1
            beststep = optimizer.res[step] == optimizer.max
            
        print("\nbeststep {}, current step {}".format(step+1, current_step+1))
        optimizer.maximize(
            init_points=0, #steps of random exploration (random starting points before bayesopt(?))
            n_iter=1, # steps of bayesian optimization
            acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
            xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
            )
    
#get best values from optimizer
n = 2**int(optimizer.max.get("params").get("n"))
steps_in= 6*int(optimizer.max.get("params").get("steps_in"))
batch_size = 2**int(optimizer.max.get("params").get("batchsize"))

inpT = int(round(optimizer.max.get("params").get("inpT")))
inpTsin = int(round(optimizer.max.get("params").get("inpTsin")))
inpSMLT = int(round(optimizer.max.get("params").get("inpSMLT")))
inpE = int(round(optimizer.max.get("params").get("inpE")))
inpSF = int(round(optimizer.max.get("params").get("inpSF")))
inpSWVL1 = int(round(optimizer.max.get("params").get("inpSWVL1")))
inpSWVL2 = int(round(optimizer.max.get("params").get("inpSWVL2")))
inpSWVL3 = int(round(optimizer.max.get("params").get("inpSWVL3")))
inpSWVL4 = int(round(optimizer.max.get("params").get("inpSWVL4")))

# correct and print best values to console
maxDict = optimizer.max
maxDict['params']['n'] = n
maxDict['params']['steps_in'] = steps_in
maxDict['params']['batchsize'] = batch_size
maxDict['params']['steps_in(days)'] = steps_in/24
print("\nBEST:\t{}".format(maxDict))


#%% Testing

#set some modeling parameters or testing
learning_rate = 1e-3
training_epochs = 100   
earlystopping_patience = 12

# check which channels were selected for final model
nchannels = 1 + inpT + inpTsin + inpSMLT + inpE + inpSF + inpSWVL1 + inpSWVL2 + inpSWVL3 + inpSWVL4
    
y_train = np.squeeze([Qnorm[steps_in:trainInds[-1]+1,]]).T
y_val = np.squeeze([Qnorm[valInds,] ]).T
y_test = np.squeeze([Qnorm[testInds,] ]).T

y_train = y_train.astype(dtype = np.float16)
y_val = y_val.astype(dtype = np.float16)
y_test = y_test.astype(dtype = np.float16)

# Reassemble data
x_intermediate = np.empty(np.shape(Pnorm) + (nchannels,),dtype='single')

x_intermediate[:,:,:,0] = Pnorm # always included

idx = 1
if inpT: 
    x_intermediate[:,:,:,idx] = Tnorm
    idx = idx+1
if inpSMLT:
    x_intermediate[:,:,:,idx] = SMLTnorm
    idx = idx+1
if inpTsin:
    x_intermediate[:,:,:,idx] = Tsinnorm
    idx = idx+1
if inpE:
    x_intermediate[:,:,:,idx] = Enorm
    idx = idx+1
if inpSF:
    x_intermediate[:,:,:,idx] = SFnorm
    idx = idx+1
if inpSWVL1:
    x_intermediate[:,:,:,idx] = SWVL1norm
    idx = idx+1
if inpSWVL2:
    x_intermediate[:,:,:,idx] = SWVL2norm
    idx = idx+1
if inpSWVL3:
    x_intermediate[:,:,:,idx] = SWVL3norm
    idx = idx+1
if inpSWVL4:
    x_intermediate[:,:,:,idx] = SWVL4norm
    idx = idx+1

x_train = np.empty((Ntrain-steps_in, steps_in, ) + np.shape(Pnorm)[1:] + (nchannels,),dtype=np.float16)
x_val = np.empty((Nval, steps_in,) + np.shape(Pnorm)[1:] + (nchannels,), dtype = np.float16)
x_test = np.empty((Ntest, steps_in,) + np.shape(Pnorm)[1:] + (nchannels,),dtype=np.float16)

#for training
for ii in range(Ntrain-steps_in):
  x_train[ii] = x_intermediate[ii:ii+steps_in]
# for validation
for ii in range(Nval):
  x_val[ii] = x_intermediate[ii + Ntrain - steps_in : ii + Ntrain]
# for testing 
for ii in range(Ntest):
  x_test[ii] = x_intermediate[ii + Ntrain + Nval + Nopt - steps_in : ii + Ntrain + Nval + Nopt]

# #convert predict/target arrays to tensors
x_train = tf.convert_to_tensor(x_train)
x_val = tf.convert_to_tensor(x_val)
x_test = tf.convert_to_tensor(x_test)
y_train = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(y_val)
y_test = tf.convert_to_tensor(y_test)

#create train/val/test datasets for model
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(Ntrain).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(Nval).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(Ntest).batch(batch_size)

#choose calculation device
with tf.device("/gpu:2"):
    
    #define early stopping callback to use in all models
    callbacks = EarlyStopping(monitor='val_loss', 
                          mode='min', 
                          verbose=1, 
                          patience = earlystopping_patience,
                          restore_best_weights = True)
    
    #generate/train model
    inimax = 10 # random number seed loop index
    testresults_members = np.zeros((Ntest, inimax))
    y_predstd = np.zeros((Ntest, inimax))
    
    for ini in range(inimax):
        model_name = 'model_ERA5_ini' + str(ini)
        print("Test: ini "+str(ini)+" of "+str(inimax-1))
        
        #generate/train model
        seed(ini+37657)
        tf.random.set_seed(ini+37657)
        
        if os.path.isdir(dir_models + '/' + model_name)==0:
        
            model = build_model(learning_rate, Pnorm, steps_in, nchannels, n)
            # model.summary()
            history = model.fit(
                train_dataset,
                validation_data = val_dataset,
                epochs = training_epochs,
                verbose = 1, 
                callbacks = [callbacks]
                )

            # plot loss during training
            pyplot.figure(figsize=(10,4))
            pyplot.title('Loss')
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='opt')
            pyplot.ylabel('Loss', size=12)
            pyplot.xlabel('Epochs',size=12)
            pyplot.legend()
            pyplot.show()

            # save model
            model.save(dir_models + '/' + model_name)
            
        else:
            model = tf.keras.models.load_model(dir_models + '/' + model_name)
            print("model loading successful")
        
        
        # sim1 = model.predict(x_test, batch_size = batch_size, verbose = 0)
        y_pred_distribution = predict_distribution(x_test, model, 100, batch_size) #based on MC_Dropout
        test_sim = Qscaler.inverse_transform(y_pred_distribution)
        
        testresults_members[:, ini], y_predstd[:, ini]= test_sim.mean(axis=1),test_sim.std(axis=1)


testresults_members_uncertainty = unumpy.uarray(testresults_members,1.96*y_predstd) #1.96 because of sigma rule for 95% confidence

# plot and save model
tf.keras.utils.plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True, dpi=300)

test_sim_mean1 = np.mean(testresults_members,axis = 1)    
sim1 = np.asarray(test_sim_mean1.reshape(-1,1))
sim1_uncertainty = np.sum(testresults_members_uncertainty,axis = 1)/inimax

Y_test_n = y_test
Y_test = Qscaler.inverse_transform(Y_test_n)
obs1 = Y_test.reshape(-1,1)

#calculate performance measures
err = sim1-obs1
err_rel = (sim1-obs1)/(np.max(Q)-np.min(Q))
NSE = nse(obs1, sim1)
try:
    r = stats.pearsonr(sim1[:,0], obs1[:,0])
except:
    r = [np.nan, np.nan]
r = r[0] #r
R2 = r ** 2
RMSE =  np.sqrt(np.mean(err ** 2))
rRMSE = np.sqrt(np.mean(err_rel ** 2)) * 100
Bias = np.mean(err)
rBias = np.mean(err_rel) * 100
scores = pd.DataFrame(np.array([[NSE, R2, RMSE, rRMSE, Bias, rBias]]),
               columns=['NSE','R2','RMSE','rRMSE','Bias','rBias'])
print(scores)

#%% Plot1 

pyplot.figure(figsize=(15,6))
sim = sim1
testresults_members = testresults_members
obs = obs1
scores = scores

y_err = unumpy.std_devs(sim1_uncertainty)

pyplot.fill_between(t[testInds,], sim.reshape(-1,) - y_err,
                    sim.reshape(-1,) + y_err, facecolor = (1,0.7,0,0.99),
                    label ='95% confidence',linewidth = 1,
                    edgecolor = (1,0.7,0,0.99))

pyplot.plot(t[testInds,], sim, color = 'r', label ="simulated mean", alpha=0.9,linewidth=1)

pyplot.plot(t[testInds,], obs, 'k', label ="observed", linewidth=1,alpha=0.3)
pyplot.title("XX Spring", size=15)
pyplot.ylabel('Q [m³/s]', size=12)
pyplot.xlabel('Date',size=12)
pyplot.legend(fontsize=12,bbox_to_anchor=(1.2, 1),loc='upper right')
pyplot.tight_layout()

s = """NSE = {:.2f}\nR²  = {:.2f}\nRMSE = {:.2f}\nrRMSE = {:.2f}
Bias = {:.2f}\nrBias = {:.2f}\n
batch_size = {:.0f}\nn = {:.0f}\nsteps_in = {:.0f}

inpP fixed
inpT = {:.0f}
inpTsin = {:.0f}
inpSMLT = {:.0f}
inpE = {:.0f}
inpSF = {:.0f}
inpSWVL1 = {:.0f}
inpSWVL2 = {:.0f}
inpSWVL3 = {:.0f}
inpSWVL4 = {:.0f}""".format(scores.NSE[0],scores.R2[0],
scores.RMSE[0],scores.rRMSE[0],scores.Bias[0],scores.rBias[0],
batch_size,n,steps_in,inpT,inpTsin,inpSMLT,inpE,inpSF,inpSWVL1,inpSWVL2,inpSWVL3,inpSWVL4)

pyplot.figtext(0.88, 0.24, s, bbox=dict(facecolor='white'))
pyplot.savefig('Test_XX_2DCNN.png', dpi=300)
pyplot.show()


#%% some(other) performance measures


testresults_members_uncertainty = unumpy.uarray(testresults_members,1.96*y_predstd) #1.96 because of sigma rule for 95% confidence
    
tf.keras.utils.plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True, dpi=300)

test_sim_mean1 = np.mean(testresults_members,axis = 1)    
sim1 = np.asarray(test_sim_mean1.reshape(-1,1))
sim1_uncertainty = np.sum(testresults_members_uncertainty,axis = 1)/inimax

Y_test_n = y_test
Y_test = Qscaler.inverse_transform(Y_test_n)
obs1 = Y_test.reshape(-1,1)

err = sim1-obs1
err_rel = (sim1-obs1)/(np.max(Q)-np.min(Q))
# NSE = nse_op(obs1, sim1, Q[refInds,])
NSE = nse(obs1, sim1)
try:
    r = stats.pearsonr(sim1[:,0], obs1[:,0])
except:
    r = [np.nan, np.nan]
r = r[0] #r
R2 = r ** 2
RMSE =  np.sqrt(np.mean(err ** 2))
rRMSE = np.sqrt(np.mean(err_rel ** 2)) * 100
Bias = np.mean(err)
rBias = np.mean(err_rel) * 100

alpha = np.std(sim1/1000)/np.std(obs1/1000)
beta = np.mean(sim1/1000)/np.mean(obs1/1000)
KGE = 1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2) #KGE

#Volume Error

Evol = 100*((np.sum(sim1)-np.sum(obs1))/np.sum(obs1))

scores = pd.DataFrame(np.array([[NSE, R2, RMSE, rRMSE, Bias, rBias, KGE,Evol]]),
                      columns=['NSE','R2','RMSE','rRMSE','Bias','rBias','KGE','Evol'])
print(scores)

print(Evol)


#%% Plot2


pyplot.figure(figsize=(10,3))
sim = sim1
testresults_members = testresults_members
obs = obs1
scores = scores

y_err = unumpy.std_devs(sim1_uncertainty)

pyplot.fill_between(t[testInds,], sim.reshape(-1,) - y_err,
                    sim.reshape(-1,) + y_err, facecolor = (1,0.7,0,0.99),
                    label ='95% confidence',linewidth = 0.8,
                    edgecolor = (1,0.7,0,0.99))


pyplot.plot(t[testInds,], sim, color = 'r', label ="simulated mean", alpha=0.8,linewidth=0.8)

pyplot.plot(t[testInds,], obs, 'k', label ="observed", linewidth=0.7,alpha=0.5)
pyplot.title("XX Spring 2D Model (ERA5-Land)", size=15)
pyplot.ylabel('Q [m³/s]', size=12)
pyplot.xlabel('Date',size=12)
pyplot.legend(fancybox = False, framealpha = 0, edgecolor = 'k')
pyplot.grid(b=True, which='major', color='#666666', alpha = 0.1, linestyle='-')
pyplot.tight_layout()
    
s = """NSE\nR²\nRMSE\nBias\nKGE"""
pyplot.figtext(0.08, 0.6, s)

s = """{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}\n""".format(scores.NSE[0],scores.R2[0],
scores.RMSE[0],scores.Bias[0],scores.KGE[0])
pyplot.figtext(0.13, 0.55, s)

pyplot.savefig('Test2D_XX_Paperplot.png', dpi=500)
pyplot.show()


#%% save results


printdf = pd.DataFrame(data=np.c_[obs,sim,y_err],index=t[testInds,])
printdf = printdf.rename(columns={0: 'Obs', 1: 'Sim', 2:'Sim_Error'})
printdf.to_csv('./results.txt',sep=';', float_format = '%.6f')

scores.to_csv('./scores.txt', sep=';',float_format='%.2f')