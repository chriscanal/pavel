
# coding: utf-8

#  ### Table of Contents Generation
#  This notebook was created with the purpose of automatically generating table of contents from text.
#  The toy dataset used to play with this problem was pulled from:
#  http://jmcauley.ucsd.edu/data/amazon/
#  
#  This dataset contains reviews of books, and titles of reviews. The input to the model will be the block of review text and the output will be the short title of the review. This should give us a decent idea of the possibilities with table content generation. The current plan for the architecture is to name any section of text based on the location of certain words in the text. So the actual input will be one hot encoded text tokens and the output will be locations of words that should be used for the ToC.

# In[1]:


import json
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from collections import Counter
from time import time
    
vocab_size = 1000
max_sequence_len = 200
max_title_len = 5

data_string = open("../Downloads/reviews_Musical_Instruments_5.json").read()
raw_data = data_string.split("\n")
data = [json.loads(d) for d in raw_data if d]


# In[ ]:


def create_long_string(obj):
    s = ""
    for example in obj:
        s += example['summary']+" "
        s += example['reviewText']+" "
    return s


# In[ ]:


def train_word_index(text, num_words):
    sequence = text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    word_counts = Counter(sequence)
    top_words = word_counts.most_common(num_words)
    word_index = dict((i+1,str(top_words[i][0])) for i in range(len(top_words)))
    word_index[0] = "UNK"
    return word_index, dict((v,k) for k,v in word_index.iteritems())


# In[ ]:


def create_one_hot_tensor(sequence,word_number_index, max_length):
    number_sequence = []
    for word in sequence:
        if word in word_number_index:
            number_sequence.append(word_number_index[word])
        else:
            number_sequence.append(0) #append 0 for the UNK word
    a = np.array(number_sequence)
    b = np.zeros((1,max_length, len(word_number_index)))
    b[0,np.arange(len(number_sequence)), a] = 1
    return b


# In[ ]:


def one_hot_tensor_to_words(tensor, number_word_index):
    sentence = ""
    for i in range(len(tensor)):
        if np.sum(tensor[i]):
            sentence += number_word_index[np.argwhere(tensor[i])[0][0]]+" "
    return sentence    


# In[ ]:


def filter_text(text, title, max_sequence_len, max_title_len):
    #Make sure that the text is in the title for
    #this base case and then make sure that the
    #review is short enough
    sequence = text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    title_seq = text_to_word_sequence(title, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    lt = len(title_seq)
    ls = len(sequence)
    if ls > max_sequence_len or lt > max_title_len or ls < 1 or lt < 1 :
        return []
    title_in_text = True
    for word in title_seq:
        if word not in sequence:
            title_in_text = False
    if title_in_text:
        return [sequence, title_seq]
    return []


# In[ ]:


def create_output_vector(text_seq, title_seq, max_sequence_len):
    output = np.zeros((1,max_sequence_len))
    num_words = len(title_seq)
    order_change_value = 0.1/num_words
    for i in range(num_words):
        output[0,text_seq.index(title_seq[i])] = 1.0 - i*order_change_value
    return output


# In[ ]:


def create_title(output_vector, input_sequence):
    copy = np.array(output_vector)
    title = ""
    while np.amax(copy) > 0.5:
        i = np.argmax(copy)
        title += input_sequence[i] + " "
        copy[i] = 0
    return title


# In[ ]:


def shape_data(data, word_number_index,max_sequence_len,max_title_len):
#     X = np.zeros((1,max_sequence_len,len(word_number_index)))
#     y = np.zeros((1,max_sequence_len))
    X = []
    y = []
    for i in range(len(data)):
        sequence = filter_text(data[i]['reviewText'], data[i]['summary'], max_sequence_len, max_title_len)
        if sequence:
            one_hot_input = create_one_hot_tensor(sequence[0],word_number_index, max_sequence_len)
            output_vector = create_output_vector(sequence[0], sequence[1], max_sequence_len)
            X.append(one_hot_input)
            y.append(output_vector)
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


# In[ ]:


text = create_long_string(data[:1000])
number_word_index, word_number_index = train_word_index(text,vocab_size-1)
start = time()
X, y = shape_data(data, word_number_index,max_sequence_len,max_title_len)
print "This took", time()-start, "seconds"


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 2
epochs = 12

# the data, split between train and test sets
x_train, y_train, x_test, y_test = X[:int(len(X)*0.8)],y[:int(len(y)*0.8)],X[int(len(X)*0.8):],y[int(len(y)*0.8):]

x_train = x_train.reshape(x_train.shape[0], max_sequence_len, vocab_size, 1)
x_test = x_test.reshape(x_test.shape[0], max_sequence_len, vocab_size, 1)
input_shape = (max_sequence_len, vocab_size, 1)

print(input_shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(max_sequence_len, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




