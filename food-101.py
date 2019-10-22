#!/usr/bin/env python
# coding: utf-8

# Food image classification using the Fast AI library and the Food-101 dataset
# =====
# The purpose of this notebook is to create a deep learning model to classify images using the Fast AI library and the Food-101 dataset. The Fast AI uses the PyTorch library. The created model will try to get close to matching some recent [state of the art results](https://arxiv.org/pdf/1612.06543.pdf). The target is >85% for top-1 accuracy. 
# 
# Dataset:  
# https://www.vision.ee.ethz.ch/datasets_extra/food-101/ 
# 
# Libraries used:  
# https://docs.fast.ai/  
# https://pytorch.org/
# 


from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
import subprocess as sp 


# Set parameters here:
# The initial batch size used is 64 for 224x224 images. ResNet-50 model is used as a starting point for the transfer learning process.
bs = 64
arch = models.resnet50
img_size = 224


path = Path('data')
#dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/food-101'
##untar_data url ended without the file extension!
#untar_data(dataset_url, 'food-101.tgz', path)


path = Path('data/food-101') 
path_img = path/'images'

train_path = 'data/food-101/train.txt'
test_path = 'data/food-101/test.txt'

def filelist2df(path):
    df = pd.read_csv(path, delimiter='/', header=None, names=['label', 'name'])
    df['name'] =  df['label'].astype(str) + "/" + df['name'].astype(str) + ".jpg"
    return df

train_df = filelist2df(train_path)
test_df = filelist2df(test_path)

ds_tfms = ([RandTransform(tfm=TfmCrop (crop_pad), kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, p=1.0, resolved={}, do_run=True, is_random=True), 
            RandTransform(tfm=TfmAffine (flip_affine), kwargs={}, p=0.5, resolved={}, do_run=True, is_random=True), 
            RandTransform(tfm=TfmCoord (symmetric_warp), kwargs={'magnitude': (-0.2, 0.2)}, p=0.75, resolved={}, do_run=True, is_random=True), 
            RandTransform(tfm=TfmAffine (rotate), kwargs={'degrees': (-10.0, 10.0)}, p=0.75, resolved={}, do_run=True, is_random=True), 
            RandTransform(tfm=TfmAffine (zoom), kwargs={'scale': (1.0, 1.1), 'row_pct': (0, 1), 'col_pct': (0, 1)}, p=0.75, resolved={}, do_run=True, is_random=True), 
            RandTransform(tfm=TfmLighting (brightness), kwargs={'change': (0.4, 0.6)}, p=0.75, resolved={}, do_run=True, is_random=True), 
            RandTransform(tfm=TfmLighting (contrast), kwargs={'scale': (0.8, 1.25)}, p=0.75, resolved={}, do_run=True, is_random=True)], 
           [RandTransform(tfm=TfmCrop (crop_pad), kwargs={}, p=1.0, resolved={}, do_run=True, is_random=True)])


data = (ImageList.from_df(df=train_df, path=path/'images', cols=1)
        .random_split_by_pct(0.2)
        .label_from_df(cols=0)
        .transform(ds_tfms, size=224)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


rows = 2
cols = 6
width = 15
height = 9




# Here is a list of the classes. We verify that there are 101 classes.

print(data.classes);
print(data.c)



top_5 = partial(top_k_accuracy, k=5)

learn = create_cnn(data, arch, metrics=[accuracy, top_5], callback_fns=ShowGraph)


# Learning rate has a huge impact on training the model. If the learning rate is too large, the loss will diverge - if it is too small, then it will take a very long time to train the model. We use the fastai learning rate finder to see what is a good starting point for a learning rate. We look for the point of the greatest negative slope on the graph. 



learn.lr_find()





epoch = 10

lr = 1e-2
learn.fit_one_cycle(epoch, slice(lr))
learn.save('food-101-test-e5')


# Let's unfreeze all the ResNet-50 weights and train them.



learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(epoch, max_lr=slice(1e-5, 1e-3))
learn.save('food-101-test-e10')
learn.fit_one_cycle(epoch, max_lr=slice(1e-6, 1e-3))
learn.save('food-101-test-e15')
learn.fit_one_cycle(epoch, max_lr=slice(1e-6, 1e-3))
learn.save('food-101-test-e20')


# So far we have been training at 224x224. Let's switch to 512x512 to increase the accuracy. Although we train to 35 epochs here, the desired accuracy on the validation set is reached at 25 epochs. 


bs=16
data = (ImageList.from_df(df=train_df, path=path/'images', cols=1)
        .random_split_by_pct(0.2)
        .label_from_df(cols=0)
        .transform(ds_tfms, size=512)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
learn = create_cnn(data, arch, metrics=[accuracy, top_5], callback_fns=ShowGraph)
learn.load('food-101-test-e20')

learn.lr_find()

# variation on learning rate
learn.unfreeze()
learn.fit_one_cycle(epoch, max_lr=slice(1e-7, 1e-2))
learn.save('food-101-test-e25-512')


# variation on learning rate
learn.unfreeze()
learn.fit_one_cycle(epoch, max_lr=slice(1e-8, 1e-3))
learn.save('food-101-test-e30-512')


# variation on learning rate
learn.unfreeze()
learn.fit_one_cycle(epoch, max_lr=slice(1e-9, 1e-4))
learn.save('food-101-test-e35-512')



interp = ClassificationInterpretation.from_learner(learn)
print(interp.most_confused(min_val=5))

# Let's load the test set and evaluate using the model. 


bs=16
test_data = (ImageList.from_df(df=test_df, path=path/'images', cols=1)
            .no_split()
            .label_from_df(cols=0)
            .transform(size=512)
            .databunch(bs=bs)
            .normalize(imagenet_stats))


learn = create_cnn(test_data, arch, metrics=[accuracy, top_5], callback_fns=ShowGraph)
learn.load('food-101-test-e30-512')




learn.validate(test_data.train_dl)


# ### Analysis
# 
# Based on the validation set, we can see a find some of the most misclassified items in the confusion matrix:
# 
# - filet mignon and steak
# - pork chop and steak
# - pancakes and pho
# - chocolate cake and chocolate mousse
# - dumplings and gyoza
# 
# #### Similar categories 
# Since filet mignon is a type of steak and gyoza is the Japanese word for dumpling, the classification between the two categories cannot be easily determined. In some examples, chocolate mousse cake does look like chocolate cake. Perhaps the categories could have been created to be more distinct. For a fixed dataset like Food-101, we can do little in our model to compensate for this. 
# 
# #### True misclassified examples
# For pancakes and pho, these are true mispredictions since they are not similar to the human eye. Pork chops and steak are more similar, but a human eye could be able to determine the difference between the two. Some more image transformations could help the model distinguish the differences. 
# 
# 
# #### Mislabeled examples
# Some of the examples are mislabeled. Based on the run of plot_multi_top_losses() above, which we can see that some of the mislabeled examples had the highest losses and highest probability of being the predicted category. We see that the labeled apple_pie image is actually a caprese_salad, the 2nd apple_pie image is actually a lasagna. The 3rd image of the apple_pie is predicted as bread_pudding - although this would be less certain for a human classifying this image. Because the dataset has mislabeled examples - there is an upper bound on the accuracy of the results - we know that 100% will not be possible. For the most part, the model is able to generalize with these mislabeled examples. 
# 
# 
# #### Other notes on the dataset
# In addition to the mislabeled examples, we see that there is quite a range of brightness and contrast through the dataset. This is meant to model actual photos that would be captured in a variety of lighting in restaurant  environments. 
# 
# #### Metrics
# Top-1 and Top-5 are used for evaluating the model as they are also used in recent [food recognition papers](https://arxiv.org/pdf/1612.06543.pdf). 
# 
# 
# #### Results on the test set:
# The test set is evaluated above on a model that has been trained on 30 epochs:  
# Top-1 87.44%   
# Top-5 97.37%  
# 
# We meet our goal of >85% for Top-1 accuracy, but we don't quite meet the state of the art results of 89.58% from WISeR. 
# 
# 
# #### Possible improvements
# 1. Only basic transformations were used in this model. Possible enhancements would be to try other augmentations like jitter or skew. Also, the brightness and contrast parameters would be tuned for a higher range. This might help with separating food that had a similar color, but different shapes - or similar shapes with different colors. 
# 2. Test time augmentation could be used - During test time, several augmentations can be used on the test set to create several predictions for each test example. The multiple results are then combined to provide one result. 
# 3. A deeper ResNet model could be used to improve the result at the cost of the size of a larger model and more computation time. 
# 
# 





