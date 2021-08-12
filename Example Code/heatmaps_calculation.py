# -*- coding: utf-8 -*-.

"""
doi of according publication [preprint]:
https://doi.org/10.5194/hess-2021-403

Contact: andreas.wunsch@kit.edu
ORCID: 0000-0002-0585-9549

https://github.com/AndreasWunsch/CNN_KarstSpringModeling/
MIT License

large parts opf the code from Sam Anderson (https://github.com/andersonsam/cnn_lstm_era)
see also: Anderson & Radic (2021): Evaluation and interpretation of convolutional-recurrent networks for regional hydrological modelling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle
from random import seed
import os
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from scipy import interpolate


#%% functions
def bayesOpt_function():
    # just a placeholder needed to load json logs
    return

#%% 
def make_heat(model, x_test, y_test, style_dict, timesteps, iters_total, iters_one_pass, verbose, tol, p_channel, batch_size):
  """
  Function by Sam Anderson (2021), slightly modified by Andreas Wunsch (2021).
  
  model: 
      keras model
  x_test:
      tf tensor; test set of ERA data, input to model (shape = Ntest x 365 x height x width x channels)
  y_test:
      tf tensor; test set of streamflow data, target output of model (shape = Ntest x 1)
  style_dict:
      dictionary: {'style' : 'RISE' or 'gauss',
                   'params' : [h,w,p_1] or sigma}
          where [h,w,p_1] are the height/width/probability of perturbation of low-res mask (for RISE algorithm); sigma is the gaussian RMS width
  timesteps:
      rangef timesteps in test set to perturb (e.g. timesteps = range(0,365) will perturb the first 365 timesteps in the test set)
  iters_total:
      number of total iterations of perturbation to do for each day in timesteps
  iters_one_pass:
      number of iterations to do at one time (typically less than iters_total for memory limits)
  verbose:
      0: print nothing
      1: print every 50th day
  tol:
      relative error threshold (when to stop pertubing model)
  batch_size:
      batchsize of the model training process
  p_channel:
      number that defines the channel that will be perturbed (others will be used as is)
  """
  #initialize heatmap as 3D numpy array: lat x lon x 1
  heat_mean = np.zeros((np.size(x_test[0,0,:,:,0]), 1))

  H = np.shape(x_test)[2] #height of input video, in pixels (latitude)
  W = np.shape(x_test)[3] #width of input video, in pixels (longitude)

  heat_prev = np.zeros((H*W,1)) #initially, the previous heatmap is 0's (for first pass)
  heat_curr = np.zeros((H*W,1)) #also initialize the current heatmap as 0's (will fill once calculated at end of first pass)
  kk = 0
  err = tol+1 #just to enter while loop
  while err > tol:
    
    print(kk)
    #loop through specified timesteps to generate mean sensitivity
    for ts in timesteps: #for each day that we will perturb

      #state progress
      if verbose:
        if np.mod(ts,iters_one_pass)==0:
          print(' Timestep ' + str(ts) + '/' + str(len(ts))) 

      #number of iterations of perturbations for one forward pass through model
      iters = iters_one_pass 

      #define perturbation: rectangular as from RISE, or gaussian 
      if style_dict['style'] == 'RISE':
    
        h = style_dict['params'][0]
        w = style_dict['params'][1]
        p_1 = style_dict['params'][2]

        x_int = np.linspace(0,W,w) #low-res x indices
        y_int = np.linspace(0,H,h) #low-res y indices

        xnew = np.arange(W)
        ynew = np.arange(H) 

        perturb_small = np.random.choice([0,1],size = (iters,1,h,w), p = [1-p_1,p_1]) #binary perturbation on coarse grid
        perturb = np.half([interpolate.interp2d(x_int,y_int,perturb_small[iter][0])(xnew,ynew) for iter in range(iters)]) #perturbation is interpolated to finer grid

      elif style_dict['style'] == 'gauss':

        sigma = style_dict['params']

        x_int = np.arange(W)
        y_int = np.arange(H)
        x_mesh, y_mesh = np.meshgrid(x_int, y_int)

        #define gaussian perturbation for each iteration being passed
        perturb = np.half([np.exp( - ( (x_mesh - np.random.randint(0,W))**2 + (y_mesh - np.random.randint(0,H))**2 ) / (2*sigma**2) ) for iter in range(iters)])

      #copy/expand dimensions of the perturbation to be added to weather video
      perturb_2D = np.copy(perturb) #the 2D perturbation for each iteration of this pass

      perturb = tf.repeat(tf.expand_dims(tf.convert_to_tensor(perturb),3),nchannels, axis = 3) #expand along channels in one image

      perturb = tf.repeat(tf.expand_dims(tf.convert_to_tensor(perturb),1),steps_in, axis = 1) #expand along images in one video

      # only perturb one channel
      mask = np.zeros((perturb.shape))
      mask[:,:,:,:,p_channel] = 1
      mask = np.half(mask)
      mask = tf.convert_to_tensor(mask)
      perturb = perturb*mask
      
      xday = x_test[ts] #current timestep in test set
      xday_iters = [xday for val in range(iters)] #repeat for each iteration (e.g. make copy for each perturbation)

      factor = np.random.choice([-1,1],p = [0.5,0.5]).astype('float16') #whether to add or subtract perturbation from input video, 50-50 chance of each
      perturb = factor*perturb

      x1 = perturb
      x2 = tf.convert_to_tensor(xday_iters)
      xday_iters_perturb = tf.math.add(x1,x2)

      x_all = tf.squeeze(tf.concat((tf.expand_dims(xday, axis = 0),xday_iters_perturb), axis = 0)) #'all' refers to original (xday) and perturbed (xday_iters_perturb)
      x_all_ds = tf.data.Dataset.from_tensor_slices(x_all).batch(batch_size = batch_size)
      y_all = model.predict(x_all_ds)

      yday = y_all[0] #first element is unperturbed model prediction
      yday_perturb = y_all[1:] #all others are perturbed model predictions for each iteration of perturbation

      ydiffs = np.abs(np.reshape(yday - yday_perturb[:iters],(-1,1))) #magnitude difference between perturbed and unperturbed streamflow
      delta = np.ones((len(ydiffs),H,W)) * ydiffs[:,None] #get dimensions to match so delta can be multiplied by perturbation

      heat_iters = np.asarray(delta[:iters]) * np.asarray(perturb_2D)
      heat = np.mean(heat_iters[:iters], axis=0) 

      heat_mean[:,0] += heat.flatten() 

      del heat, heat_iters, delta, ydiffs, x_all, xday_iters #delete for memory

    n_iters = iters_one_pass*(kk+1)
    heat_curr = np.copy(heat_mean) / n_iters
    err = np.mean(np.abs(heat_curr - heat_prev)) / np.mean(heat_prev)

    heat_prev = np.copy(heat_curr)

    kk += 1

  heat_mean = heat_mean /(iters_total * len(timesteps))

  return heat_mean


#%% Set directories and load data
dir_data = './data_pickle' #where to save trained model outputs
dir_models = './Results' #where to save trained model outputs
dir_output = './heatmaps'

# os.chdir(dir_output)

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

#years/indices of testing/training
#modify accordingly
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

refInds = np.squeeze(np.argwhere((t.year<testStartYear)))
                                 
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


#%% Define Bayesian Optimization to be able to load from existing logs:

pbounds = {'steps_in': (1,10*4),
           'n': (7,7),
           'batchsize': (7,7),
           'inpTsin': (0,1),
           'inpSMLT': (0,1), 
           'inpE': (0,1),
           'inpT': (0,1),
           'inpSF': (0,1),
           'inpSWVL1': (0,1),
           'inpSWVL2': (0,1),
           'inpSWVL3': (0,1),
           'inpSWVL4': (0,1)}

optimizer = BayesianOptimization(
    f= bayesOpt_function, 
    pbounds=pbounds, 
    random_state=1, 
    verbose = 0 
    )

# #load existing optimizer
log_already_available = 0
if os.path.isfile("./logs.json"):
    load_logs(optimizer, logs=["./logs.json"]);
    print("\nExisting optimizer is already aware of {} points.".format(len(optimizer.space)))
    log_already_available = 1
    
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


#%% Compile test data
learning_rate = 1e-3
training_epochs = 100   
earlystopping_patience = 12

nchannels = 1 + inpT + inpTsin + inpSMLT + inpE + inpSF + inpSWVL1 + inpSWVL2 + inpSWVL3 + inpSWVL4

y_train = np.squeeze([Qnorm[steps_in:trainInds[-1]+1,]]).T
y_val = np.squeeze([Qnorm[valInds,] ]).T
y_test = np.squeeze([Qnorm[testInds,] ]).T

y_train = y_train.astype(dtype = np.float16)
y_val = y_val.astype(dtype = np.float16)
y_test = y_test.astype(dtype = np.float16)

x_intermediate = np.empty(np.shape(Pnorm) + (nchannels,),dtype='single')
x_intermediate[:,:,:,0] = Pnorm
channel_names = ['P']
idx = 1
if inpT:
    x_intermediate[:,:,:,idx] = Tnorm
    channel_names.append('T')
    idx = idx+1
if inpSMLT:
    x_intermediate[:,:,:,idx] = SMLTnorm
    channel_names.append('SMLT')
    idx = idx+1
if inpTsin:
    x_intermediate[:,:,:,idx] = Tsinnorm
    channel_names.append('Tsin')
    idx = idx+1
if inpE:
    x_intermediate[:,:,:,idx] = Enorm
    channel_names.append('E')
    idx = idx+1
if inpSF:
    x_intermediate[:,:,:,idx] = SFnorm
    channel_names.append('SF')
    idx = idx+1
if inpSWVL1:
    x_intermediate[:,:,:,idx] = SWVL1norm
    channel_names.append('SWVL1')
    idx = idx+1
if inpSWVL2:
    x_intermediate[:,:,:,idx] = SWVL2norm
    channel_names.append('SWVL2')
    idx = idx+1
if inpSWVL3:
    x_intermediate[:,:,:,idx] = SWVL3norm
    channel_names.append('SWVL3')
    idx = idx+1
if inpSWVL4:
    x_intermediate[:,:,:,idx] = SWVL4norm
    channel_names.append('SWVL4')
    idx = idx+1

x_train = np.empty((Ntrain-steps_in, steps_in, ) + np.shape(Tnorm)[1:] + (nchannels,),dtype=np.float16)
x_val = np.empty((Nval, steps_in,) + np.shape(Tnorm)[1:] + (nchannels,), dtype = np.float16)
x_test = np.empty((Ntest, steps_in,) + np.shape(Tnorm)[1:] + (nchannels,),dtype=np.float16)

#training
for ii in range(Ntrain-steps_in):
  x_train[ii] = x_intermediate[ii:ii+steps_in]
# #validation
for ii in range(Nval):
  x_val[ii] = x_intermediate[ii + Ntrain - steps_in : ii + Ntrain]
# #testing ()
for ii in range(Ntest):
  x_test[ii] = x_intermediate[ii + Ntrain + Nval + Nopt - steps_in : ii + Ntrain + Nval + Nopt]

# #convert target arrays to tensors
x_train = tf.convert_to_tensor(x_train)
x_val = tf.convert_to_tensor(x_val)
x_test = tf.convert_to_tensor(x_test)
y_train = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(y_val)
y_test = tf.convert_to_tensor(y_test)

#%% Load existing Models and calculate heatmaps
with tf.device("/gpu:2"): # adapt to your available device
    
    for c in range(nchannels): #perturb one channel at a time
    
        inimax = 10 # use 10 different trained models
        heat = np.zeros((T.shape[1]*T.shape[2],inimax)) # preallocate
        print(channel_names[c])
        for ini in range(inimax):
            
            fileName = dir_output + '/heatmap_'+channel_names[c]+'_channel_ini'+str(ini)+'.csv'
            if os.path.isfile(fileName): # check for previous calculation runs to save time
                temp_load = pd.read_csv(fileName,header=None)
                temp = np.asarray(temp_load[0])
            else:
                print("Model: ini "+str(ini)+" of "+str(inimax-1))
                seed(ini+37657)
                tf.random.set_seed(ini+37657)
                
                model_name = 'model_ERA5_ini' + str(ini)# + '.h5'
                model = tf.keras.models.load_model(dir_models + '/' + model_name)
                print("model loading successful")
        
                #parameters for heatmaps
                sigma = 1.5 #radius of perturbation
                style_dict = {'style' : 'gauss', #style of perturbation: gaussian (not RISE/rectangular)
                              'params' : sigma}
                timesteps_heat = range(Ntest) #timesteps in test set to perturb
                iters_total = 200 #total iterations of perturbation
                iters_one_pass = 50 #number of iterations to pass through model at one time (too high --> RAM issues)
                tol = 5e-3
                
                # heatmap
                temp = make_heat(model = model,
                                 x_test = x_test, 
                                 y_test = y_test, 
                                 style_dict = style_dict, 
                                 timesteps = timesteps_heat, 
                                 iters_total = iters_total, 
                                 iters_one_pass = iters_one_pass, 
                                 verbose = 0,
                                 tol = tol,
                                 p_channel = c, # channel to pertubate, other are left as is
                                 batch_size = batch_size
                                 )
                 
            heat[:,ini] = temp.reshape(-1,)
            
            fileName = 'heatmap_'+channel_names[c]+'_channel_ini'+str(ini)+'.csv'
            np.savetxt(dir_output + '/' + fileName, temp.reshape(-1,), delimiter = ',')

        heat_mean = np.mean(heat,axis=1)
        
        #save mean heatmap
        fileName = 'heatmap_'+channel_names[c]+'_channel.csv'
        np.savetxt(dir_output + '/' + fileName, heat_mean, delimiter = ',')
        
#save channel_names file for enx script: plot heatmaps
np.savetxt(dir_output+'/channelnames.txt', channel_names,fmt='%s', delimiter = ',')
