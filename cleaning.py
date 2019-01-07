# Code to clean data and collect 20K images

#!/anaconda3/bin/python
import time
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import constants
import pickle


def get_labels_images_train():
    labels_to_use = os.path.join('statistics/sorted_common_labels.csv') 
    labels_200_df = pd.read_csv(labels_to_use)[:200]
    labels_200_values= labels_200_df['label_code'].tolist()
    
    train_human_labels = pd.read_csv(constants.TRAIN_HUMAN_IMGLABELS)
    train_human_labels = train_human_labels[train_human_labels['Confidence'] == 1]
    print(train_human_labels.shape)
    train_human_labels = train_human_labels[train_human_labels['LabelName']\
                                            .isin(labels_200_values)]
    print(train_human_labels.head())
    #Human training labels where Confidence = 1 and 
    #labels are in the 200 most common labels
    print(train_human_labels.shape)
    load_start_dict = time.time()
    #Pickle the training images with 200 labels ---
    with open('training_dict.pickle','rb') as total_labels:
        total_human_labels_dict = pickle.load(total_labels)
    
    print('Training_dict loaded in {}'.format(time.time() - load_start_dict))
    
    begin = time.time()
    train_human_200labels = os.path.join(constants.INPUT_DIR,
                                           'human_200labels_dict.pickle')
    if os.path.exists(train_human_200labels):
        print('Pickle file of image IDs with the 200labels already exists')
        with open('human_200labels_dict.pickle', 'rb') as handle:
                train_human_200labels_dict = pickle.load(handle)
        print(len(train_human_200labels_dict))
    else:
        train_human_200labels_dict=  {}
        print('Creating a new pickle file...')
        #For each unique imageID, make a label dict
        for id in train_human_labels['ImageID'].unique():
                if id in total_human_labels_dict.keys():
                    train_human_200labels_dict[id] = []
                
                    for label in total_human_labels_dict[id]:
                        
                        if label in labels_200_values:
                            train_human_200labels_dict[id].append(label)
                    
                    
        train_human_200labels_dict = dict( [(k,v) for k,v in
                                            train_human_200labels_dict.items() if len(v)>0])
        with open('human_200labels_dict.pickle', 'wb') as handle:
            pickle.dump(train_human_200labels_dict, handle,\
                       protocol=pickle.HIGHEST_PROTOCOL)
    print(time.time() - begin)
    print('Pickling complete..')
   
    # from these, take only the first 20k images
    train_human_labels_20k = train_human_labels[:75000]
    print(train_human_labels_20k.shape)
    print(train_human_labels_20k['ImageID'].unique().shape)
    # (24770,) unique images
    
    images_20k = train_human_labels_20k['ImageID'].unique().tolist()
    #print(images_20k[:10])

def get_labels_images_test():
    labels_to_use = os.path.join('statistics/sorted_common_labels.csv')
    labels_200_df = pd.read_csv(labels_to_use)[:200]
    print(labels_200_df.head())
    test_labels = pd.read_csv(constants.TUNING_LABELS,
                              names=['ImageID','label_codes'])
    #Choose the test tuning labels that are in labels_to_use
    test_labels['label_codes'] =(test_labels['label_codes']).map(lambda x:\
                                                                 x.split())

    test_tuning_200labels =\
    os.path.join(constants.INPUT_DIR,'test_tuning_200labels_dict.pickle')

    if os.path.exists(test_tuning_200labels):
        print('Test_tuning pickle file already exists')
        with open('test_tuning_200labels_dict.pickle','rb') as handle:
                test_tuning_dict = pickle.load(handle)
        print(len(test_tuning_dict))
        print(test_tuning_dict)
    else:
        test_tuning_200labels_dict = {}

        for row in test_labels.itertuples(index=True, name='Pandas'):
            if set(getattr(row,
                           'label_codes')).issubset(labels_200_df['label_code']\
                                                    .tolist()):
                test_tuning_200labels_dict[getattr(row, 'ImageID')] = \
                        getattr(row, 'label_codes')
        with open('test_tuning_200labels_dict.pickle', 'wb') as handle:
            pickle.dump(test_tuning_200labels_dict, handle,\
                       protocol=pickle.HIGHEST_PROTOCOL)
        print('Test Pickling complete.')
        print(len(test_tuning_200labels_dict)) 
    

if __name__ == '__main__':
    get_labels_images_train()
    get_labels_images_test()
