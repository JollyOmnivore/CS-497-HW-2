import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import math
import pandas as pd


GMEDF = pd.read_csv("GME.csv")
x = np.arange(0, len(GMEDF))
y = GMEDF.iloc[:, 1] 

modelSine = Sequential([
    Dense(1,activation='linear',input_shape=(1,)),
    Dense(8200,activation='ReLU'),
    Dense(1,activation='linear')
    ]
)

#not very cool just makes sure it built the way I want
modelSine.summary()

#compile model and mean error 
modelSine.compile(loss='mse',optimizer='adam',metrics=['mae'])


#train  time
#epochs in times trained
modelSine.fit(x,y, epochs=5000,batch_size=8, verbose=1)

#predict
output = modelSine.predict(x)

print(x.shape)
print(output.shape)

plt.plot(x,y,'b',x,output,'r')
plt.show()



    