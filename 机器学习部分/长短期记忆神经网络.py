from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from keras.models import Sequential

#parameters for LSTM
nb_lstm_outputs = 30  #神经元个数
nb_time_steps = 28  #时间序列长度
nb_input_vector = 28 #输入序列


#data preprocessing: tofloat32, normalization, one_hot encoding
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
 #build model
model = Sequential()
model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
model.add(Dense(10, activation='softmax'))

#compile:loss, optimizer, metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train: epcoch, batch_size
model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1)

model.summary()
 
score = model.evaluate(x_test, y_test,batch_size=128, verbose=1)
print(score)



