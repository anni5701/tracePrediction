import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statistics
from visualization import *



class PredictionModel:

    BATCH_SIZE = 100 
    INPUT_DIM = 6
    OUTPUT_DIM = 1 
    RNN_HIDDEN_DIM = 32

    EPCOHS = 50
    TIME_PERIODS = 5

    def __init__(self, dataframe):
        x_train, y_train, x_test, y_test = self.prepare_data(dataframe)   
        print("The trainings data is of shape {} and the test data is of shape {}". format(x_train.shape, x_test.shape))

        neg_train, pos_train = np.bincount(y_train[:,0].astype(int))
        neg_test, pos_test = np.bincount(y_test[:,0].astype(int))
        print('Train data:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(len(y_train), pos_train, 100 * pos_train / len(y_train)))
        print('Test data:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(len(y_test), pos_test, 100 * pos_test / len(y_test)))

        # initial_bias = np.log([(pos_train + pos_test) / (neg_train + neg_test)]) # used for imbalanced data

        model = self.create_lstm(input_shape=x_train.shape[1]) # , output_bias=initial_bias

        print ('Fitting model...')
        
        history = model.fit(x_train, y_train ,batch_size= self.BATCH_SIZE , epochs=self.EPCOHS,  validation_split = 0.2, verbose = "2") #, shuffle= "batch"

        plot_accuracy(history)
        plot_loss(history)

        predictions = self.predictions_test_data(model, x_test, y_test)

        plot_confusion_matrix(predictions, y_test)

        
        

    def predictions_test_data(self, model, x_test, y_test):
        print ('Use  mmodel on test data...')
        predictions = model.predict(x_test)
        predictions = np.mean(predictions, axis=1)
        predictions = np.where(predictions <= 0.5, 0, 1) 
        print("The accuracy on the test data is {}".format(accuracy_score(y_true=y_test, y_pred=predictions)))
        return predictions
        

    def create_segments_and_labels(self, dataframe: pd.DataFrame , step: int, label_name: str, n_features: int):
        print ('Create segments and labels...')
        segments = []
        labels = []
        for i in range(0, len(dataframe) - step, step):
            xs = dataframe['vel_x'].values[i: i + step]
            ys = dataframe['vel_y'].values[i: i + step]
            zs = dataframe['vel_z'].values[i: i + step]
            nav_x = dataframe['nav_ax'].values[i: i + step]
            nav_y = dataframe['nav_ay'].values[i: i + step]
            nav_z = dataframe['nav_az'].values[i: i + step]
            # Retrieve the most often used label in this segment
            label = statistics.mode(dataframe[label_name][i: i + step])
            segments.append([xs, ys, zs, nav_x, nav_y, nav_z])
            labels.append(label)

        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, step, n_features)
        labels = np.asarray(labels)

        return reshaped_segments, labels

    def prepare_data(self,df, test_split = 0.2, segment_split= False):
        print ('Split into test and trainings data...')
        
        df['touch_bin'] = df['touch'].apply(lambda x : 0 if x <= 0.5 else 1)

        split = int(len(df) * (1- test_split))
        train_data = df.iloc[:split]
        test_data =  df.iloc[split:]

        x_train, y_train = self.create_segments_and_labels(train_data, self.TIME_PERIODS, 'touch_bin', self.INPUT_DIM)
        x_test, y_test = self.create_segments_and_labels(test_data, self.TIME_PERIODS, 'touch_bin', self.INPUT_DIM)

        num_time_periods, n_features = x_train.shape[1], x_train.shape[2]

        input_shape = (num_time_periods*n_features) # num segments times num features
        x_train = x_train.reshape(x_train.shape[0], input_shape) # is shape[0] the number of samples?
        y_train = y_train.reshape(-1,1)

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        num_time_periods, n_features = x_test.shape[1], x_test.shape[2]
        input_shape = (num_time_periods*n_features)
        x_test = x_test.reshape(x_test.shape[0], input_shape) # is shape[0] the number of samples?
        y_test = y_test.reshape(-1,1)

        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')


        return x_train, y_train, x_test, y_test




    def create_lstm(self,input_shape, input_dim = INPUT_DIM, output_bias= None):
        print ('Create the LSTM model...')

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        model = Sequential()
        model.add(tf.keras.layers.Reshape((self.TIME_PERIODS, input_dim), input_shape=(input_shape,)))
        model.add(tf.keras.layers.LSTM(units = 128, input_shape = (input_shape,))) # return_sequences = True,
        #model.add(tf.keras.layers.LSTM(units = 10, return_sequences = True))
        model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid', bias_initializer=output_bias)) 
        model.compile('adam', 'binary_crossentropy', 'accuracy') #metrics=[tf.keras.metrics.BinaryAccuracy()]
        print(model.summary())
        return model



