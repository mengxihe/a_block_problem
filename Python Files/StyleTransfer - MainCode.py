#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


# In[2]:


import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
# define size of matplitlib figures
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


# In[3]:


# function to transform tensor into new images
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# In[4]:


# paths from example with labrador and kandinsky image

#content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
#style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


# In[5]:


# change to your path

#content_path = 'C:\\Users\\chris\\Documents\\ABK SS 21\\Computional Explorations\Assignment 5\\tiles5.jpg'
#style_path = 'C:\\Users\\chris\\Documents\\ABK SS 21\\Computional Explorations\Assignment 5\\floorplan3.jpg'

content_path = 'D:\\_ITECH - S2\\COMP EXPLORATIONS\\Assignment 5 - DL Project\\StyleTransfer\\_BarcelonaContentImage.jpg'
style_path = 'D:\\_ITECH - S2\\COMP EXPLORATIONS\\Assignment 5 - DL Project\\StyleTransfer\\StyleImage_Picasso.jpg'

# In[6]:


def load_img(path_to_img):
    max_dim = 512
    
    #load in image, decode jpg and convert to floating tensor
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    print(img.shape)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    #calculate new size dimensions
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    
    #resize original image
    img = tf.image.resize(img, new_shape)
    #add dimension to the beginning (why?)
    img = img[tf.newaxis, :]
    return img
    


# In[7]:


def imshow(image, title=None):
    #remove added dimension again (why?)
    if len(image.shape)  > 3:
        image = tf.squeeze(image, axis=0)
        
    plt.imshow(image)
    if title:
        plt.title(title)


# In[8]:


#using above defined funtions
content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1,2,1)
imshow(content_image, 'tiles / content')

plt.subplot(1,2,2)
imshow(style_image, 'barcelona grid / style')

print(content_image.shape)
print(content_image[0,0,0,0]*255)


# In[9]:


#tutorial link
#https://www.tensorflow.org/tutorials/generative/style_transfer

#architecture dataset
#https://www.kaggle.com/wwymak/architecture-dataset


# In[10]:


#adapt input for vgg19 pretrained model
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape


# In[11]:


predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob)in predicted_top_5]


# In[12]:


vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
    print(layer.name)


# In[13]:


content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# In[14]:


def vgg_layers(layer_names):
    #load the model. Load pretraines VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top= False, weights= 'imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    
    model = tf.keras.Model([vgg.input], outputs)
    return model


# In[15]:


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statisctics of each layer's output
for name, output in zip(style_layers, style_outputs):
    print(name)
    print(" shape:", output.numpy().shape)
    print(" min:", output.numpy().min())
    print(" max:", output.numpy().max())
    print(" mean:", output.numpy().mean())
    print()


# In[16]:


def gram_matrix(input_tensor):
    results = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return results/(num_locations)


# In[17]:


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    #maybe mistake? __call__
    def __call__(self, inputs):
    #def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        
        content_dict = {content_name: value
                       for content_name, value
                       in zip(self.content_layers, content_outputs)}
        
        style_dict = {style_name: value
                     for style_name, value
                     in zip(self.style_layers, style_outputs)}
        
        return {'content': content_dict, 'style': style_dict}
        


# In[18]:


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("   ", name)
    print("   shape: ", output.numpy().shape)
    print("   min: ", output.numpy().min())
    print("   max: ", output.numpy().max())
    print("   mean: ", output.numpy().mean())
    print()
    
print('Contents:')
for name, output in sorted(results['content'].items()):
    print("   ", name)
    print("   shape: ", output.numpy().shape)
    print("   min: ", output.numpy().min())
    print("   max: ", output.numpy().max())
    print("   mean: ", output.numpy().mean())


# In[19]:


style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


# In[20]:


image = tf.Variable(content_image)


# In[21]:


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# In[22]:


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


# In[23]:


# weights change

#original weights from tutorial:

#style_weight=1e-2 
#content_weight=1e4


style_weight = 1e10
content_weight = 1e3


# In[24]:


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    
    return x_var, y_var


# In[25]:


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


# In[26]:


out_path = 'D:\\_ITECH - S2\\COMP EXPLORATIONS\\Assignment 5 - DL Project\\StyleTransfer\\Training4\\'

# In[27]:


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    
    #custom
    tf.print(tf.squeeze(style_loss), output_stream=('file://' + out_path + 'logs/style_loss.txt'))
   
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                                           for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers  
    #custom
    tf.print(tf.squeeze(content_loss), output_stream=('file://' + out_path + 'logs/content_loss.txt'))
    
    loss = style_loss + content_loss
    #custom
    tf.print(tf.squeeze(loss), output_stream=('file://' + out_path + 'logs/style_content_loss.txt'))
    
    return loss


# In[28]:


tf.image.total_variation(image).numpy()


# In[29]:


#changeable
total_variation_weight=50


# In[30]:


#os.mkdir(out_path + 'logs/')

if not os.path.exists(out_path + 'logs/'):
    os.makedirs(out_path + 'logs/')


# In[31]:


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        variation_loss = total_variation_weight*tf.image.total_variation(image)
        tf.print(tf.squeeze(variation_loss), output_stream=('file://' + out_path + 'logs/variation_loss.txt'))
        
        loss += variation_loss
        tf.print(tf.squeeze(loss), output_stream=('file://' + out_path + 'logs/loss.txt'))
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# In[32]:


image = tf.Variable(content_image)


# In[ ]:


train_step(image)
train_step(image)
train_step(image)
train_step(image)
train_step(image)

display.display(tensor_to_image(image))

#tensor_to_image(image).save(out_path+'syle_' + str(style_weight)+'_' + 'content_'+str(content_weight)+'_'+'my_nice_image'+'.jpeg')


# In[ ]:


import time
start = time.time()


#set epochs
epochs = 50
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end=' ', flush=True)
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print("Train step: {}".format(step))
    
    tensor_to_image(image).save(out_path+'syle_' + str(style_weight)+'_' + 'content_'+str(content_weight)+'_'+'epoch_'+str(step)+'.jpeg')
    
end = time.time()
print("Total time: {:.1f}".format(end-start))


# In[ ]:


#https://www.tensorflow.org/tutorials/generative/style_transfer

