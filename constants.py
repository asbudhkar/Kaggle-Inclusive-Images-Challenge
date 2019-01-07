#!/anaconda3/bin/python
import os
IMAGE_DIR = './train/'
IMAGE_DIR_20k = './train/train_20k'
IMAGE_DIR_60k = './train/train_60k'
SOURCE = './'
IMG_SIZE = 64
STATISTICS = 'statistics/'
INPUT_DIR = './'
CLASSES_TRAINABLE = os.path.join(INPUT_DIR, 'classes-trainable.csv')
CLASS_DESCRIPTIONS = os.path.join(INPUT_DIR, 'class-descriptions.csv')
TRAIN_HUMAN_IMGLABELS = os.path.join(INPUT_DIR, 'train-annotations-human-imagelabels-boxable.csv')
TUNING_LABELS = os.path.join(INPUT_DIR, 'tuning_labels.csv')
SAVED_MODELS = os.path.join(INPUT_DIR, 'models')
TRAIN_200LABEL_PICKLE = os.path.join(SOURCE,'human_200labels_dict.pickle')
TEST_200LABEL_PICKLE = os.path.join(SOURCE, 'test_tuning_200labels_dict.pickle')
SORTED_TRAINING_LABELS = os.path.join(STATISTICS, 'sorted_common_labels.csv')
SORTED_TEST_LABELS = os.path.join(STATISTICS, 'sorted_tuning_labels.csv')