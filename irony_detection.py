import numpy as np
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization

class text_data:
    
    """
    Processes the training data sets, which is a list of lists.
    Each row in the txt file is a list in the form of ['Tweet index','Label','Tweet text'].
    Takes care of tokenization and other feature engineerings.
    """
    
    def __init__(self, train_data):
        self.data = train_data
        self.length = len(self.labels())
        self.vocabulary_size = len(self.tokenizer().word_index)

    def texts(self, label=None):
        '''
        Generates the sentences for training.
        
        IN:  label:  None to output all texts; 0 / 1 to output the corresponding class.
        OUT:  List of sentences(text)
        '''
        if label:
            X = [re.sub(r'http[s]?://[\S]+', '' ,i[2]) for i in self.data if int(i[1]) == label]
        else:
            X = [re.sub(r'http[s]?://[\S]+', '' ,i[2]) for i in self.data]
        return X

    def labels(self):
        """
        Generates the coorsponding labels.
        1 - Sarcastic, 0 - Non-sarcastic
        
        OUT:  List of labels(int)
        """        
        y = np.array([int(i[1]) for i in self.data])
        return y
    
    def tokenizer(self):
        """
        Generates a tokenizer and fits the original text for preprocessing.
        
        OUT:  A Tokenizer object fit with the training texts.
        """
        tokenizer = Tokenizer(filters='!"$%&()*+,-.:;<=>?@[\\]^_`{|}~\t\n',
                             oov_token='unk')
        tokenizer.fit_on_texts(self.texts())
        return tokenizer
    
    def text_sequences(self, tokenizer=None, max_sent_length=200):
        """
        Generates an encoded matrix for the texts.
        
        IN:  
            tokenizer: None for the training data, and the tokenizer of the training data for the test data.
            max_sent_length: Max length of sentence for padding.
        OUT:
            An numpy array.
        """
        if not tokenizer:
            tokenizer = self.tokenizer()
        seq = tokenizer.texts_to_sequences(self.texts())
        
        seq = sequence.pad_sequences(seq, maxlen=max_sent_length)
        return seq
    
    def add_extra_features(self, *extra_features):
        
        """
        Adds new features to the texts sequences.
        
        IN:
            ndarrays with the same number of rows as the text_sequences.
        OUT:
            New ndarrays with the added features on the left.
        """
        seq = self.text_sequences()
        for feat in extra_features:
            if feat.shape[0] != self.length:
                raise ValueError('Size of array does not match')
            seq = np.hstack((feat, seq))
        return seq 
    
def open_parse_file(x):
    
    """
    Opens the training data sets file as a list of lists.
    Each row in the txt file is a list in the form of ['Tweet index','Label','Tweet text']
    """
        
    with open(x, encoding='utf-8') as f:
        training_set = f.readlines()
    for n,i in enumerate(training_set):
        training_set[n] = i[:-1].split('\t')
    return training_set[1:]


def build_model3(n_words=5000, 
                embedding_length=20, 
                n_neuron_lstm=50, 
                max_sent_length=200, 
                dropout_rate=0.25,
                n_neuron_elu=5,
                regularization = None
               ):

    """
    Builds the LSTM model.
    
    IN:
        n_words: Size of the vocabulary(INT)
        embedding_length:  Layer output dimension(INT)
        n_neuron_lstm:  Number of neurons in the LSTM layer(INT)
        max_sent_length:  Maximum length of a sentense(INT)
        dropout_rate:  Dropout rate for the LSTM layer(FLOAT between 0 and 1)
        n_neuron_elu:  Number of neurons in the elu layer(INT)
        regularization:  regularization for the elu layer
        
    OUT:
        Keras Sequential model    
    """
    
    model = Sequential()
    model.add(Embedding(n_words, embedding_length, input_length=max_sent_length))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(n_neuron_lstm))
    model.add(Dense(n_neuron_elu, activation='elu', kernel_regularizer=regularization))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


training_file = open_parse_file('dataset/train/SemEval2018-T3-train-taskA_emoji_ironyHashtags.txt')
test_file = open_parse_file('dataset/test/SemEval2018-T3_gold_test_taskA_emoji.txt')

train = text_data(training_file)
test = text_data(test_file)

model = build_model(n_words=a.vocabulary_size, embedding_length=8, n_neuron_lstm=5, n_neuron_elu=3, regularization=l1(0.02), dropout_rate=0.6)
model.fit(train.text_sequences(), train.labels(), epochs=8, batch_size=64, validation_split=0.15)

scores = model.evaluate(test.text_sequences(tokenizer=a.tokenizer()), test.labels(), verbose=1)
print("loss:{0:.2f}, acc: {1:.2f}%".format(scores[0],scores[1]*100))

model.save('best_model.m1')
