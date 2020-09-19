



import string
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from pickle import load
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences

max_length = 34



def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


filename = "C:/Users/VT/Desktop/Flickr Dataset/Flickr dataset/Flickr8k.token.txt"
doc = load_doc(filename)


def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping



descriptions = load_descriptions(doc)


def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)



clean_descriptions(descriptions)


def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


filename = 'C:/Users/VT/Desktop/Flickr Dataset/Flickr dataset/Flickr_8k.trainImages.txt'
train = load_set(filename)



def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions



train_descriptions = load_clean_descriptions('C:/Users/VT/Desktop/Flickr Dataset/Flickr dataset/descriptions.txt', train)

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1


def greedySearch(photo, model):
    in_text = 'startseq'        
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        photo = np.resize(photo, (1,2048))
        yhat = model.predict([photo,sequence], verbose=0)        
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def get_predictions():
    model = tf.keras.models.load_model('C:/Users/VT/Desktop/Imagecaptiongenerator/model_4.h5')
# model.load_weights('C:/Users/Laxmi/Desktop/Image captioning/model_4.h5')

    images = 'C:/Users/VT/Desktop/Flickr Dataset/Flicker8k_Dataset_Image/'
    with open("C:/Users/VT/Desktop/Flickr Dataset/Flickr dataset/encoded_test_images.pkl", "rb") as encoded_pickle:
        encoding_test = load(encoded_pickle)
    index = np.random.choice(1000)
    pic = list(encoding_test.keys())[index] 
    image = encoding_test[pic].reshape((1,2048))
    x=plt.imread(images+pic)
    st.sidebar.image(x,use_column_width=True)  
    caption=greedySearch(image,model)
    st.write('## Caption Generated is: ')
    st.write(caption)

st.title('Image caption generator')
# if st.button('Generate a Random Image'):

st.sidebar.markdown('## Input Image')
if st.button('Generate a Random Image'):
    get_predictions()

      
    
