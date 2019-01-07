# DenseNet pipeline

# Linear algebra
import numpy as np

# Data preprocessing
import pandas as pd 

import tensorflow as tf

# Load and store dat
import pickle

from sklearn.utils import shuffle

# Image operations

import cv2

import os

import sklearn

# Save plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# DenseNet model
from ./models/dense_net import *

# Train images directory
images_dir_name = './train'

# Test images directory
test_images_dir_name = './stage_1_test_images'

# Common labels
classes_trainable= pd.read_csv('statistics/sorted_common_labels.csv',engine='python'
all_labels=classes_trainable['label_code']
all_labels=all_labels[:200]
print ('The number of unique labels is {}'.format(len(all_labels)))
num_labels = len(all_labels)
labels_index = {label:idx for idx, label in enumerate(all_labels)}

# Retrieve the list of train images 
train_image_names = [img_name[:-4] for img_name in os.listdir(images_dir_name)]
test_image_names = [img_name[:-4] for img_name in os.listdir(test_images_dir_name)]
train_image_names = train_image_names[:20000]
print('loaded training images')
print ("number of training images is {}".format(len(train_image_names)))

# Retrieve the list of train labels labels
print('read labels..')

batch_size = 64
input_size = 20000
# Build dictionary
train_images = []
train_labels_raw = []
labels_raw=[]
count = 0

training_dict = {}

# Load training data
with open('human_200labels_dict.pickle', 'rb') as handle:
    training_dict = pickle.load(handle)
print("Load finished")

for x, y in training_dict.items():
  train_images.append(x)
  list=[]
  for label in y:
    try:
       list.append(labels_index[label])
    except:
       continue
  train_labels_raw.append(list)
 
# Multi-hot encoding
def multi_hot_encode(x, num_classes):
    encoded = []
    for labels in x:
        labels_encoded = np.zeros(num_classes)
        for item in labels:
            labels_encoded[item] = 1
        encoded.append(labels_encoded)
    encoded = np.array(encoded)
    return encoded
train_labels = multi_hot_encode(train_labels_raw, 200)

# Random Flip
def flip(x):
    return np.fliplr(x)

# Normalize
def normalize(x):
    return (x.astype(float) - 128)/128

# Define the dimensions of the processed image
x_dim = 100
y_dim = 100
n_channels = 3

# Scaling for logic for an image data
def scale(x):
    	return cv2.resize(x, (x_dim, y_dim)) 

# Read and pre-process image
def preprocess(image_name):
    img = cv2.imread(image_name)
    if img is not None:
        scaled = scale(img)
        flipped=flip(scaled)
        normalized = normalize(flipped)
        return np.array(normalized)

# Build the generator for training
def generator(samples, sample_labels, batch_size=32):
    num_samples = len(samples)
    
    while 1: 
        samples,sample_labels=sklearn.utils.shuffle(samples, sample_labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_labels = sample_labels[offset:offset+batch_size]
            images = []
            labels = []
            for i, batch_sample in enumerate(batch_samples):
                image = preprocess(images_dir_name+'/'+batch_sample+'.jpg')
                images.append(image)
                labels.append(batch_labels[i])
            X_train = np.array(images)
            y_train = np.array(labels)
            return X_train,y_train

# Build the generator for training
def test_generator(samples, sample_labels, batch_size=32):
    num_samples = len(samples)

    while 1: 
        sklearn.utils.shuffle(samples, sample_labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_labels = sample_labels[offset:offset+batch_size]
            images = []
            labels = []
            for i, batch_sample in enumerate(batch_samples):
                image = preprocess(test_images_dir_name+'/'+batch_sample+'.jpg')
                images.append(image)
                labels.append(batch_labels[i])
            X_test = np.array(images)
            y_test = np.array(labels)
            return X_test,y_test

from sklearn.model_selection import train_test_split
Xtrain, Xvalid, ytrain, yvalid = train_test_split(train_images, train_labels, test_size=0.01)
images=[]
labels=[]

for i,sample in enumerate(Xvalid):
    image=preprocess(images_dir_name+'/'+sample+'.jpg')
    images.append(image)
    labels.append(yvalid[i])
X_test=np.array(images)
y_test=np.array(labels)

# Placeholder for image
x = tf.placeholder("float", [None,100,100,3])

# Placeholder for training flag
training_flag = tf.placeholder(tf.bool)

# Placeholder for labels
y1_ = tf.placeholder("float", [None,200])

# Learning rate
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

# Load DenseNet model
y1 = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).model

print("Model loaded")

# Sigmoid cross entropy for multilabel classification
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y1,labels=y1_))

# Optimizer
opt = tf.train.MomentumOptimizer(0.01, 0.9)
train_op = opt.minimize(cross_entropy)

sess = tf.Session()

# Calculate Correct prediction
correct_prediction=tf.equal(tf.round(y1), y1_)

# Calculate Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()

def metric_variable(shape, dtype, validate_shape=True, name=None):

    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES],
        validate_shape=validate_shape,
        name=name,
    )

# F2 score calculation
def streaming_counts(y_true, y_pred, num_classes):
# Weights for the weighted f1 score
    weights1 = metric_variable(
        shape=[num_classes], dtype=tf.float64, validate_shape=False, name="weights"
    )
    # Counts for the macro f1 score
    tp_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="tp_mac"
    )
    fp_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fp_mac"
    )
    fn_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fn_mac"
    )
    # Counts for the micro f1 score
    tp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="tp_mic"
    )
    fp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fp_mic"
    )
    fn_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fn_mic"
    )
    #   - Update ops for the macro f1 score
    up_tp_mac = tf.assign_add(tp_mac, tf.count_nonzero(y_pred * y_true, axis=0))
    up_fp_mac = tf.assign_add(fp_mac, tf.count_nonzero(y_pred * (y_true - 1), axis=0))
    up_fn_mac = tf.assign_add(fn_mac, tf.count_nonzero((y_pred - 1) * y_true, axis=0))

    #   - Update ops for the micro f1 score
    up_tp_mic = tf.assign_add(
        tp_mic, tf.count_nonzero(y_pred * y_true, axis=None)
    )
    up_fp_mic = tf.assign_add(
        fp_mic, tf.count_nonzero(y_pred * (y_true - 1), axis=None)
    )
    up_fn_mic = tf.assign_add(
        fn_mic, tf.count_nonzero((y_pred - 1) * y_true, axis=None)
    )
    # Update op for the weights, just summing
    up_weights = tf.assign_add(weights1, tf.reduce_sum(y_true, axis=0))

    # Grouping values
    counts = (tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights1)
    updates = tf.group(up_tp_mic, up_fp_mic, up_fn_mic, up_tp_mac, up_fp_mac, up_fn_mac, up_weights)

    return counts, updates


def stream_f1(counts):
    
    tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights= counts

    # normalize weights
    weights /= tf.reduce_sum(weights)

    # computing the micro f2 score
    prec_mic = tp_mic / (tp_mic + fp_mic)
    rec_mic = tp_mic / (tp_mic + fn_mic)
    f1_mic = 5 * (prec_mic * rec_mic / (4*prec_mic + rec_mic))
    f1_mic = tf.reduce_mean(f1_mic)

    # computing the macro and wieghted f2 score
    prec_mac = tp_mac / (tp_mac + fp_mac)
    rec_mac = tp_mac / (tp_mac + fn_mac)
    f1_mac = 5 * (prec_mac * rec_mac / (4*prec_mac + rec_mac))
    f1_wei = tf.reduce_sum(f1_mac * weights)
    f1_mac = tf.reduce_mean(f1_mac)

    return f1_mic, f1_mac, f1_wei

def tf_f1_score(y_true, y_pred):

    f1s = [0, 0, 0]
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        f1s[i] = tf.reduce_mean(f1)
    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)
    f1s[2] = tf.reduce_sum(f1 * weights)
    micro, macro, weighted = f1s
    return micro, macro, weighted

tf_f1 = tf_f1_score(y1_, y1)
y_true=tf.cast(y1_,tf.float64)
y_false=tf.cast(y1,tf.float64)
counts, update = streaming_counts(y_true, y_false, 200)
f1 = stream_f1(counts)

# Store losses 
losses_train=[]
losses_val=[]
e_train=[]
acc_train=[]
e_val=[]
acc_val=[]

# Run graph
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    for j in range (50):
        training_loss=0
        training_acc=0
        loss_all=0
        acc_all=0
        for i in range (0, input_size, batch_size):
            X_train,y_train=generator(Xtrain,ytrain,64)
           
            feed_dict={
                x: X_train,
                y1_: y_train,
                training_flag:True,
                learning_rate: 0.01}

            # Batch loss 
            _,c=sess.run([train_op,cross_entropy], feed_dict=feed_dict)
            loss_all+=c

            # Batch accuracy 
            acc,_=sess.run([accuracy,update], feed_dict=feed_dict)
            acc_all+=acc           
            print(acc)
            
            if i % 512 == 0:
               print("training on image #%d" % i)
        
        # Epoch loss
        training_loss=loss_all/(int(input_size/batch_size))
        losses_train.append(training_loss)
        
        # Epoch accuracy
        training_acc=acc_all/(int(input_size/batch_size))
        acc_train.append(training_acc)
        e_train.append(j)
        print("Epoch ",j)
        mic, mac, wei = [f.eval() for f in f1]
        print("{:40}".format("\nStreamed, batch-wise f1 scores:"), mic, mac, wei)
    
    # Testing         
    for i in range (0, 2000, batch_size):
        if i + batch_size < 2000:
            
            feed_dict={
                x: X_test,
                y1_:y_test,
                training_flag:True,
                learning_rate:0.01}    
            acc,_=sess.run([accuracy,update],feed_dict=feed_dict)
            print("Accuracy",acc)
    mic, mac, wei=[f.eval() for f in f1]
    print("\n", mic, mac, wei)

# Plot Loss vs Epochs
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(e_train,losses_train,'r-')
plt.xlabel('Epochs')
plt.ylabel('Losses')
fig.savefig('train_loss.png')
fig = plt.figure()

# Plot Accuracy vs Epochs

ax = fig.add_subplot(111)
plt.plot(e_train,acc_train,'r-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
fig.savefig('train_acc.png')
