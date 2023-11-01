#!/usr/bin/env python
# coding: utf-8

# In[74]:


import os

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

import tensorflow as tf


# In[75]:


#change to input path with images
#import_path = 'C:/Users/chris/OneDrive/Dokumente/jupyter_notebooks_style_transfer/training11/'
import_path = 'D:/_ITECH - S2/COMP EXPLORATIONS/Assignment 5 - DL Project/StyleTransfer/Training3/'
logs = 'logs/'

filename_loss = 'loss.txt'
#filename_style_content_loss = 'style_content_loss.txt'
filename_style_loss = 'style_loss.txt'
filename_content_loss = 'content_loss.txt'
filename_variation_loss = 'variation_loss.txt'

file_path_loss = os.path.join(import_path, logs, filename_loss)
#file_path_style_content_loss = os.path.join(import_path, logs, filename_style_content_loss)
file_path_style_loss = os.path.join(import_path, logs, filename_style_loss)
file_path_content_loss = os.path.join(import_path, logs, filename_content_loss)
file_path_variation_loss = os.path.join(import_path, logs, filename_variation_loss)


loss_array = np.loadtxt(file_path_loss)
loss_array = loss_array[:-5]

#style_content_loss_array = np.loadtxt(file_path_style_content_loss)
#style_content_loss_array = style_content_loss_array[:-5]

style_loss_array = np.loadtxt(file_path_style_loss)
style_loss_array = style_loss_array[:-5]

content_loss_array = np.loadtxt(file_path_content_loss)
content_loss_array = content_loss_array[:-5]

variation_loss_array = np.loadtxt(file_path_variation_loss)
variation_loss_array = variation_loss_array[:-5]


step_array = np.arange(0, len(loss_array), 1)


print(loss_array.shape)
print(style_loss_array.shape)
print(content_loss_array.shape)

print(len(loss_array))
print(step_array)


# In[76]:


import glob

list_of_images = glob.glob(import_path + '*.jpeg')
latest_file = max(list_of_images, key=os.path.getctime)
print(latest_file)


# In[77]:


endpth = os.path.basename(latest_file)
print(endpth)


# In[78]:


style_weight = endpth.split('_')[1]
content_weight = endpth.split('_')[3]

step_count = endpth.split('_')[5]
step_count = step_count.split('.')[0]

print("{}\n{}\n{}".format(style_weight, content_weight, step_count))


# In[79]:




fig = plt.figure()
bx = fig.add_subplot()
fig.subplots_adjust(top=1)

bx.plot(step_array, loss_array, label = 'loss')
#plt.plot(step_array, style_content_loss_array, label = 'style_content_loss')
#plt.plot(step_array, style_loss_array, label = 'style_loss')
#plt.plot(step_array, content_loss_array, label = 'content_loss')

bx.set_xlabel('training steps')
bx.set_ylabel('loss')

bx.set_ylim([0, 1.9e10])

bx.set_title('full loss graph')
bx.text(len(step_array) + (len(step_array)*0.1), 1.5e10, 'Style weights = ' + str(style_weight))
bx.text(len(step_array) + (len(step_array)*0.1), 1.4e10, 'Content weights = ' + str(content_weight))

plt.legend()

fig.savefig(import_path + logs +'full_loss_graph.png', dpi=300,  bbox_inches='tight')


# In[80]:


start_cut = 0
end_cut = len(step_array)-0

loss_array_start = loss_array[start_cut:end_cut]
#style_content_loss_array_start = style_content_loss_array[start_cut:end_cut]
style_loss_array_start = style_loss_array[start_cut:end_cut]
content_loss_array_start = content_loss_array[start_cut:end_cut]
variation_loss_array_start = variation_loss_array[start_cut:end_cut]

step_array_start = step_array[start_cut:end_cut]


# In[81]:


plt.plot(step_array_start, loss_array_start, label='loss')
#plt.plot(step_array_start, style_content_loss_array_start, label='style_content')
plt.plot(step_array_start, style_loss_array_start, label='style loss')
plt.plot(step_array_start, content_loss_array_start, label='content loss')
plt.plot(step_array_start, variation_loss_array_start, label='variation loss')

plt.xlabel('training steps')
plt.ylabel('loss')

y_limit = 5e8

axes = plt.gca()
axes.set_ylim([0, y_limit])

plt.legend()


# In[82]:


fig = plt.figure()
ax = fig.add_subplot()
fig.subplots_adjust(top=1)

ax.plot(step_array_start, loss_array_start, label='loss')
#ax.plot(step_array_start, style_content_loss_array_start, label='style_content')
ax.plot(step_array_start, style_loss_array_start, label='style loss')
ax.plot(step_array_start, content_loss_array_start, label='content loss')
ax.plot(step_array_start, variation_loss_array_start, label='variation loss')

ax.set_xlabel('training steps')
ax.set_ylabel('loss')

ax.set_title('loss graph')
ax.text(11000, (y_limit-(y_limit/10)), 'Style weights = ' + str(style_weight))
ax.text(11000, (y_limit-(y_limit/10)*1.5), 'Content weights = ' + str(content_weight))

#axes = plt.gca()
#axes.set_ylim([0, 5e7])

ax.set(ylim=(0, y_limit))

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
ax.figsize=(2440*px, 1080*px)


ax.legend()
fig.savefig(import_path + logs +'zoom1_loss_graph.png', dpi=300,  bbox_inches='tight')


# In[83]:


fig = plt.figure()
cx = fig.add_subplot()
fig.subplots_adjust(top=1)

new_limit = y_limit * 0.1

cx.plot(step_array_start, loss_array_start, label='loss')
#cx.plot(step_array_start, style_content_loss_array_start, label='style_content')
cx.plot(step_array_start, style_loss_array_start, label='style loss')
cx.plot(step_array_start, content_loss_array_start, label='content loss')
cx.plot(step_array_start, variation_loss_array_start, label='variation loss')

cx.set_xlabel('training steps')
cx.set_ylabel('loss')

cx.set_title('loss graph')
cx.text(11000, (new_limit-(new_limit/10)), 'Style weights = ' + str(style_weight))
cx.text(11000, (new_limit-(new_limit/10)*1.5), 'Content weights = ' + str(content_weight))

#axes = plt.gca()
#axes.set_ylim([0, 5e7])

cx.set(ylim=(0, new_limit))

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
cx.figsize=(2440*px, 1080*px)


cx.legend()
fig.savefig(import_path + logs +'zoom2_loss_graph.png', dpi=300,  bbox_inches='tight')


# In[ ]:




