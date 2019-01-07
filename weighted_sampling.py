# Code to get sampled data

import numpy as np
import os,time
from collections import Counter
from ./constants import constants 
import random

def weighted_sampling(inputs, weights, num_of_samples):
    sum_of_weights = sum(weights)
    samples = []
    weighted_samples = []

    for sample in range(num_of_samples):
        samples.append(random.random() * sum_of_weights)
    samples = sorted(samples)

    sample_idx = 0
    current_weight = 0

    for i,w in enumerate(weights):
        while sample_idx < num_of_samples and current_weight + w > \
              samples[sample_idx]:
            weighted_samples.append(inputs[i])
            sample_idx+=1
        current_weight +=w
    
    return outputs

def get_weights(df, col_name, max_scale = 100):
    counts = df[col_name].values
    counts = np.sqrt(counts)
    v_min = np.min(counts)
    v_max = np.max(counts)
    counts = (counts - v_min)/(v_max - v_min)
    counts = 1 - counts
    counts = ((counts * max_scale) + 1)/ max_scale
    return counts


def get_weights_by_counts(counts, num_of_labels = 200):
    
    counts = np.array(counts)
    total = np.sum(counts)
    weights = (total/(num_of_labels*counts))
    return weights 

def get_weighted_sample(df, num_of_samples):
    weights = get_weights_by_counts(df['label_counts'].values)

    return weighted_sampling(df['ImageID'].values, weights, num_of_samples)


if __name__=='__main__':

       # From entire training dictionary collect weighted sampled data with 200 labels
       with open('training_dict.pickle','rb') as total_labels:
                 total_human_labels_dict = get_weighted_sample(pickle.load \
                                                               (total_labels))
                 with open ('human_labels_200_dict.pickle', 'wb') as final:
                     pickle.dump(total_human_labels_dict, final)

