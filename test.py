import json, ast
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from unidecode import unidecode

with open('email_response.json','r') as f:
    inp = json.load(f)

inp = ast.literal_eval(json.dumps(inp))
#print inp

data = {'mail' : [], 'reply': []}
for key in inp:
        mails = inp[key]
	for r in range(1,len(mails)):
		data['mail'].append(mails[0])
		data['reply'].append(mails[r])


#print (data['mail'])

# data['sentiment'] = ['pos' if (x>3) else 'neg' for x in data['stars']]
#data['mail'] = data['mail'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
#data['reply'] = data['reply'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

# for idx,row in data.iterrows():
#     row[0] = row[0].replace('rt',' ')
        
#data['mail'] = [x.encode('ascii') for x in data['mail']]
#data['reply'] = [x.encode('ascii') for x in data['reply']]
print (data['mail'])

tokenizer1 = Tokenizer(num_words=2500, lower=True,split=' ')
tokenizer1.fit_on_texts(data['mail'])
#print(tokenizer1.word_index)  # To see the dicstionary
X = tokenizer1.texts_to_sequences(data['mail'])
X = sequence.pad_sequences(X)

tokenizer2 = Tokenizer(num_words=2500, lower=True,split=' ')
tokenizer2.fit_on_texts(data['reply'])
# #print(tokenizer.word_index)  # To see the dicstionary
Y = tokenizer2.texts_to_sequences(data['reply'])
Y = sequence.pad_sequences(X)

embed_dim = 128
lstm_out = 200
batch_size = 32

model = Sequential()
#print X[200].shape
model.add(Embedding(2500, embed_dim,input_length = X[1].shape[0]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(lstm_out, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(562,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#print(model.summary())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 36)

#Here we train the Network.

model.fit(X_train, Y_train, batch_size =batch_size, epochs = 1,  verbose = 5)

score,acc = model.evaluate(X_test,Y_test,verbose = 2,batch_size = batch_size)
print ("Score = ",score," accuracy = ",acc)
