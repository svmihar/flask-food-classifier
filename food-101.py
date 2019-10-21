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


# Set parameters here:
# The initial batch size used is 64 for 224x224 images. ResNet-50 model is used as a starting point for the transfer learning process.
bs = 64
arch = models.resnet50
img_size = 224


# Grab the dataset - uncomment the line to actually download the dataset and untar it
path = Path('data')
dataset_url = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
#untar_data(dataset_url, 'food-101.tar.gz', path)


path = Path('data/food-101') #untar creates food-101 subdirectory
path_img = path/'images'


# ## Food-101 dataset
# The Food-101 dataset has 101 food categories, with 1000 images provided for each category for a total of 101,000 images. The Food-101 dataset provides a list of examples for training as well as another list for testing. Since the dataset specifies an equal amount of examples for each category, we do not need to worry about class imbalances. For each category, 250 images are in the test set and 750 for the training set. The training set is split into 20% for the validation set and 80% for the training set. 
# 
# Training set: 60600 images  
# Validation set: 15150 images  
# Test set: 25250 images  
# 
# The validation set is used to check the performance of the model after each training epoch. The test set is evaluated only once at the end of this notebook to provide a final accuracy score. 
# 
# The examples for each category are located in subdirectories with the category names. The file list is treated like a csv file, though now delimited with a '/' symbol. This is read using Pandas into a Pandas DataFrame structure and then is modified so that the paths and .jpg file extensions are added. Fast AI provides a way of reading the DataFrame structure and indicating that the labels are in column 0 and the examples are in column 1. 
# 
# The images are resized to 224x224 for faster training of the model. Data augmentation during training uses the default set of transforms provided by the Fast AI library as well as the default parameters. These are provided by the get_transforms() method. Since it's difficult to see what parameters and transforms are used from this method, the extracted list and parameters are provided below(ds_tfms). The pixel values of the examples are also normalized based on the ImageNet values. These examples could have been normalized by determining the mean & standard deviation of this specific dataset, but it appears that the ImageNet values were sufficient for a good result. 


train_path = 'data/food-101/meta/train.txt'
test_path = 'data/food-101/meta/test.txt'

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


# Let's visualize some examples in the dataset - these are with transformations applied:

data.show_batch(rows=3, figsize=(10, 10))

# Show one original image and then show several versions of that same image with the transformations applied. This helps to visualize what the transforms are doing. 



img = open_image('data/food-101/images/apple_pie/157083.jpg')
img.show()


rows = 2
cols = 6
width = 15
height = 9


[img.apply_tfms(get_transforms()[0]).show(ax=ax) for i,ax in enumerate(plt.subplots(
    rows,cols,figsize=(width,height))[1].flatten())];


# Here is a list of the classes. We verify that there are 101 classes.


print(data.classes);
print(data.c)


# For metrics, we will look at the accuracy(top-1) and the top-5 accuracy(notated as top_k_accuracy). ResNet-50 and the its weights are used as a starting point for transfer learning. ResNet-50 is selected as it is a fairly  The Fast AI library then discards the classfication layer of the ResNet-50 model and then attaches a few additional layers. More information can be found [in the fastai documentation](https://docs.fast.ai/vision.learner.html#create_cnn)


top_5 = partial(top_k_accuracy, k=5)

learn = create_cnn(data, arch, metrics=[accuracy, top_5], callback_fns=ShowGraph)


# Learning rate has a huge impact on training the model. If the learning rate is too large, the loss will diverge - if it is too small, then it will take a very long time to train the model. We use the fastai learning rate finder to see what is a good starting point for a learning rate. We look for the point of the greatest negative slope on the graph. 



learn.lr_find()




learn.recorder.plot(suggestion=True)


# We train the model for 5 epochs at a time, occasionally reducing the learning rate. We are training only the last few layers that was added for transfer learning. 



lr = 1e-2
learn.fit_one_cycle(5, slice(lr))
learn.save('food-101-test-e5')


# Let's unfreeze all the ResNet-50 weights and train them.



learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-3))
learn.save('food-101-test-e10')
learn.fit_one_cycle(5, max_lr=slice(1e-6, 1e-3))
learn.save('food-101-test-e15')
learn.fit_one_cycle(5, max_lr=slice(1e-6, 1e-3))
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
learn.recorder.plot(suggestion=True)


# variation on learning rate
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-7, 1e-2))
learn.save('food-101-test-e25-512')


# variation on learning rate
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-8, 1e-3))
learn.save('food-101-test-e30-512')


# variation on learning rate
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-9, 1e-4))
learn.save('food-101-test-e35-512')


# We can visually look at some of the mispredictions and see if there is any conclusion that can be drawn from them. fastai has built in functions to display examples based on top losses, high/low probability. The confusion matrix is also useful to see what is commonly misclassified.


interp = ClassificationInterpretation.from_learner(learn)


interp.plot_top_losses(9, figsize=(15, 11))



interp.most_confused(min_val=5)



interp.plot_multi_top_losses()



interp.plot_confusion_matrix(figsize=(20, 20), dpi=200)


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





