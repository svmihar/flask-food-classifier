#!/usr/bin/env python
# coding: utf-8

# # Training a Food Image Classfier

# In this notebook, we will train a Resnet-50 CNN on the [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) dataset.
#
# The dataset included 101 food categories with each class having 1000 images.

# ## Imports

# Import the fastai library and other dependencies

# In[1]:


import pandas as pd
import requests
import glob
from io import BytesIO
import numpy as np
import os
import shutil
import pprint
import json


# In[2]:


from fastai import *
from fastai.vision import *


# In[ ]:





# # Retrieving Data

# Run the below commands  if you need to fetch the data.
#

# In[ ]:


get_ipython().system('mkdir -p ../data')
get_ipython().system('wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz -P ../data')
get_ipython().system('tar xzf ../data/food-101.tar.gz -C ../data')


# # Load Data

# The data is stored in the "data" folder; one directory above.
# Each of the 101 food categores is stored in its own folder

# In[3]:


get_ipython().system('ls ../data')


# In[4]:


get_ipython().system('ls ../data/food-101/')


# The first 5 of the 101 categories

# In[5]:


get_ipython().system('ls ../data/food-101/images | head -n 5')


# read the images

# In[6]:


path = Path('../data/food-101/images')


# In[7]:


data = ImageDataBunch.from_folder(path, valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=8, bs=64).normalize(imagenet_stats)


# In[8]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# ## Train Model

# # Initialize Rest

# In[9]:


learn = create_cnn(data, models.resnet34, metrics=error_rate,pretrained=True)


# In[10]:


learn.lr_find()
learn.recorder.plot()


# In[11]:


#learn = create_cnn(data, models.resnet34, metrics=error_rate, model_dir="../../prod")
#learn.model = torch.nn.DataParallel(learn.model)


# In[12]:


lr = 1e-2


# In[13]:


learn.fit_one_cycle(8 , lr)


# In[ ]:





# In[14]:


model_name="resnet34"


# In[15]:


learn.save(f'{model_name}-stage-1')


# In[16]:


learn.load(f'{model_name}-stage-1')


# In[17]:


learn.unfreeze()


# In[18]:


learn.lr_find(start_lr=1e-09, end_lr=1e-3)


# In[19]:


learn.recorder.plot(skip_end=10)


# In[20]:


learn.fit_one_cycle(5, max_lr=slice(1e-8,1e-4))


# In[21]:


learn.save(f'{model_name}-stage-2')


# In[22]:


learn.load(f'{model_name}-stage-2');


# # Interpretation

# In[23]:


learn.load(f'{model_name}-stage-2');


# In[24]:


interp = ClassificationInterpretation.from_learner(learn)


# In[25]:


interp.plot_top_losses(9, figsize=(15,11))


# In[26]:


interp.plot_confusion_matrix(figsize=(50,50), dpi=30)


# In[27]:


interp.most_confused(min_val=2)


# In[28]:


final_model_name = f'{model_name}-final'


# In[29]:


learn.save(final_model_name)


# In[ ]:





# # Testing on Different Data

# The model

# ### Loading Trained model

# In[30]:


learn.load(final_model_name);


# In[31]:


learn.data.classes


# In[32]:


data2 = ImageDataBunch.single_from_classes(path, data.classes
                                           , tfms=get_transforms()
                                           , size=224).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34)
learn.load(final_model_name)


# In[33]:


data2.classes, data2.c


# In[ ]:





# **bibimbap**

# In[34]:


bibimbap_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Dolsot-bibimbap.jpg/1200px-Dolsot-bibimbap.jpg"


# In[ ]:





# In[35]:


#!wget {url} -O sample.jpg


# In[36]:


url = bibimbap_url


# In[37]:


def fetch_image(url):
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    return img


# In[ ]:





# In[38]:


img = fetch_image(bibimbap_url)
pred_class,pred_idx,outputs = learn.predict(img)
pred_class , pred_idx, outputs


# In[39]:


def predict(url):
    img = fetch_image(url)
    pred_class,pred_idx,outputs = learn.predict(img)
    res =  zip (learn.data.classes, outputs.tolist())
    predictions = sorted(res, key=lambda x:x[1], reverse=True)
    top_predictions = predictions[0:5]
    pprint.pprint( top_predictions)
    return img.resize(500)


# **Baby Back Rib**

# In[40]:


baby_back_url ="https://upload.wikimedia.org/wikipedia/commons/e/ee/Baby_back_ribs_with_fries.jpg"


# In[41]:


predict(baby_back_url)


# **Cat**

# In[42]:


cat_image_url = "https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492__480.jpg"


# In[43]:


predict(cat_image_url)


# In[ ]:





# **Icecream**

# In[44]:


icecream_url = "https://upload.wikimedia.org/wikipedia/commons/3/31/Ice_Cream_dessert_02.jpg"


# In[45]:


predict(icecream_url)


# **Banana**

# In[46]:


banana_url = "https://upload.wikimedia.org/wikipedia/commons/d/de/Bananavarieties.jpg"


# In[47]:


predict(banana_url)


# In[ ]:





# # Prepare for production

# To make our model available as a web app, we will need to save:
# - final model
# - list of class names

# In[ ]:





# Remove existing model artifacts

# In[51]:


shutil.rmtree("../models",ignore_errors=True)


# Copy the models stored locally to folder above

# In[85]:





# In[94]:


final_model_directory = os.getcwd()+ "/../models"
final_model_name='model.pkl'


# In[95]:


learn.export(final_model_directory+f"/{final_model_name}")


# In[ ]:





# In[87]:


get_ipython().system('pwd')


# Save the list of classes

# In[67]:


with open('../models/classes.txt', 'w') as f:
    json.dump(learn.data.classes,f)


# In[54]:


get_ipython().system('pwd')


# In[ ]:





# # Load

# In[96]:


learn3= load_learner(final_model_directory,final_model_name)


# In[100]:


img


# In[99]:


learn3.predict(img)


# ## Next Steps

# Refer to this [guide](https://github.com/npatta01/food-classifier/blob/master/docs/2_heroku_app.md) to deploy the model on heroku

# In[ ]: