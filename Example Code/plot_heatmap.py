# -*- coding: utf-8 -*-
"""
@author: Andreas Wunsch
"""

import numpy as np
import pickle
from matplotlib import pyplot
import os
import geopandas as gpd
from matplotlib import cm
import matplotlib

#%% Start - Load Data

dir_data = './'
dir_models = './'
dir_output = dir_models

os.chdir(dir_output)

# load one pickle file for coordinates and data dimensions
pickle_in = open(dir_data + '/' + 'TDict_ERA5land_large_1981-2018_daily.pickle','rb')
tempDict = pickle.load(pickle_in)
T = np.asarray(tempDict['T'])

#load file which includes channel names
channel_names = np.genfromtxt(dir_output+'/channelnames.txt',dtype='str')
import matplotlib.ticker as plticker

#%%
for channel in channel_names:
    fileName = 'heatmap_'+channel+'_channel.csv'
    heat_mean = np.loadtxt(dir_output + '/' + fileName, delimiter=',')
    
    fontsze = 16
    
    #load Shapefile
    shape = './catchment_shapefile.shp'
    catchment_shape = gpd.read_file(shape)
    
    shape_eu = './coastline_shapefile.shp'
    europe = gpd.read_file(shape_eu)
    
    #color scaling
    vmax = np.max(np.abs(heat_mean))
    vmin = np.min(np.abs(heat_mean))
    
    #coordinates from data Dict
    lon = tempDict['lon']
    lat = tempDict['lat']

    lat1 = np.array([lat,]*lon.shape[0]).transpose()
    lon1 = np.array([lon,]*lat.shape[0])
    
    #plotting
    fig, ax = pyplot.subplots(nrows = 1, ncols = 1, figsize = (7,7))
    im = ax.pcolormesh(np.asarray(lon1), np.asarray(lat1), heat_mean.reshape(np.shape(T[0])), shading='nearest',cmap = 'RdBu_r',vmin = vmin, vmax = vmax)
    
    catchment_shape.plot(ax=ax,color = 'k',linewidth = 2.5)
    europe.plot(ax=ax,color = 'k',alpha = 0.15)
    ax.set_aspect('equal')
    
    #ticking
    loc = plticker.MultipleLocator(base=0.3) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    
    loc = plticker.MultipleLocator(base=0.3) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    
    #plot customization
    ax.tick_params(labelsize = fontsze-4)
    ax.set_xlabel('Longitude [$^\circ$E]', fontsize = fontsze)
    ax.set_ylabel('Latitude [$^\circ$N]', fontsize = fontsze)
    
    ax.set_title('XX Spring, '+channel+' Channel', fontsize = fontsze+1, fontweight = 'bold')
        
    ax.set_xlim((np.min(lon)-0.05, np.max(lon)+0.05))
    ax.set_ylim((np.min(lat)-0.05, np.max(lat)+0.05))
    
     #colorbar
    cax = fig.add_axes([0.92, 0.21, 0.07, 0.6]) #colourbar gets own axis
    mapname = 'RdBu'
    cmap = cm.get_cmap(mapname)
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    scalarMap = cm.ScalarMappable(norm = norm, cmap = cmap)
    cb = fig.colorbar(im, cax = cax, orientation = 'vertical',format='%.1e')
    pyplot.ylabel('$S(x,y)$', fontsize = fontsze-4)#, cax)
    
    # save png
    pyplot.savefig('heatmap_XX_'+channel+'_channel.png', dpi = 300, bbox_inches = 'tight')
    pyplot.show()