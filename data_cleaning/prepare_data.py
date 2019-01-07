# Code to create training dictionary from train images

images_dir_name = '../train'
input_dir = '../'

# Retrieve all the labels and store those into a collection
classes_trainable = pd.read_csv(input_dir+'classes-trainable.csv')
all_labels = classes_trainable['label_code']
print ('The number of unique labels is {}'.format(len(all_labels)))

# Build the index dictionary based on the labels collection
labels_index = {label:idx for idx, label in enumerate(all_labels)}
train_image_names = train_image_names[:20000]
print ("number of training images is {}".format(len(train_image_names)))
labels = pd.read_csv('./train-annotations-human-imagelabels-boxable.csv', engine='python')
train_images = []
train_labels_raw = []
labels_raw=[]
count = 0
training_dict = {}
# Save in dictionary
for index, row in labels.iterrows():
    if row[0] in train_image_names:
        if row[0] in training_dict:
                training_dict[row[0]].append(row[2])
        else:
                training_dict[row[0]] = [row[2]]
                count+=1
# Save dictionary in file                
with open('../training_dict.pickle', 'wb') as handle:
    pickle.dump(training_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Load finished")