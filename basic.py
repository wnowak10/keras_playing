import numpy as np
import matplotlib.pylab as plt
x=np.linspace(0,100,100) #0-100 linearly
y=x>50 # define output pretty decisively!
y=1*y
# plt.plot(x,y)
# plt.show()

a=np.random.rand(100) # noise
# plt.plot(a,y)
# plt.show()

# convert y to one hot
n_values=np.max(y) + 1       
y=np.eye(n_values)[y]

# source activate py35
# data=np.zeros([100,1,2])
data=np.zeros([100,2,1])

data[:,0,0]=x
data[:,1,0]=a


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, SGD

num_features=2
model = Sequential()
model.add(LSTM(10,input_shape=[num_features,1]))
# model.add(LSTM(16,input_dim=1))
model.add(Dense(2))
model.add(Activation('softmax'))
adam=SGD()
model.compile(	loss='categorical_crossentropy',
				optimizer=adam,
				metrics=["accuracy"])


h=model.fit(data, # train data
			y, # train labels, in one hot
			batch_size=50, 
			nb_epoch=300, # how many epochs to train 
			verbose=1 ) # show progress

x=model.predict_classes(data)

# plot accuracy over training epochs
plt.plot(h.history['acc'])
plt.show()