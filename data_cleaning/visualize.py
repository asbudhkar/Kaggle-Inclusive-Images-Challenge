# Create subsets of the original data to 20k and 60k images

#!/anaconda3/bin/python

import numpy as np
import pandas as pd
import os
import random, time
from ./constants import constants


ALL_IMAGES = os.path.join(constants.IMAGE_DIR)
TWENTYK_IMAGES = os.path.join(ALL_IMAGES, 'train_20k/')
SIXTYK_IMAGES = os.path.join(ALL_IMAGES, 'train_60k/')
tuning_df = pd.read_csv(constants.TUNING_LABELS, names =['id','labels'] )
tuning_counts = tuning_df['labels'].str.split().apply(pd.Series).stack().value_counts()
tuning_counts_df = pd.DataFrame(tuning_counts, columns = ['label_counts'])
attributes_df = pd.read_csv(constants.CLASS_DESCRIPTIONS, index_col ='label_code')

tuning_counts_df = tuning_counts_df.join(attributes_df).sort_values('label_counts',ascending=False)
tuning_counts_df.index.name= 'label_code'

print('\nThe most common labels are:\n')

print(tuning_counts_df.head())
tuning_counts_df.to_csv('statistics/sorted_tuning_labels.csv')
classes = pd.read_csv(constants.CLASSES_TRAINABLE)['label_code'].values.tolist()
human_labels_df = pd.read_csv(constants.TRAIN_HUMAN_IMGLABELS,index_col = ['LabelName'])
human_labels_df = human_labels_df[human_labels_df.index.isin(classes)].join(attributes_df).sort_values('Confidence', ascending=False,)
label_counts = human_labels_df.index.value_counts()
label_counts_df = label_counts.to_frame(name='label_counts')
label_counts_df.index.name= 'label_code'

print('\n\n Get the 10 top labels:\n\n')

label_counts_sorted = (label_counts_df.join(attributes_df).sort_values('label_counts',ascending=False))
label_counts_sorted.to_csv('statistics/sorted_common_labels.csv')
