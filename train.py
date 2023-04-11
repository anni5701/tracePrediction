import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
import os
import scipy



class PredictionModel:

    BATCH_SIZE = 100 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
    INPUT_DIM = 6
    OUTPUT_DIM = 1 
    RNN_HIDDEN_DIM = 32
    DROPOUT_RATIO = 0.1 # proportion of neurones not used for training



    #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
    # would a list with indices also work?
    EPCOHS = 50
    TIME_PERIODS = 10

    def __init__(self, dataframe):
        x_train, y_train= self.prepare_data(dataframe)    #, X_test, y_test 
        model = self.create_lstm(input_shape=x_train.shape[1]) 

        print ('Fitting model...')
        
        history = model.fit(x_train, y_train,batch_size= self.BATCH_SIZE ,class_weight=None, epochs=self.EPCOHS,  validation_split = 0.2, verbose = "1")
        self.create_plots(history)

        

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
            #label = scipy.stats.mode(df[label_name][i: i + step])[0][0]
            label = np.average(dataframe[label_name][i: i + step])
            if label <= 0.5:
                label = 0
            else:
                label = 1
            segments.append([xs, ys, zs, nav_x, nav_y, nav_z])
            labels.append(label)

        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, step, n_features)
        labels = np.asarray(labels)

        return reshaped_segments, labels

    def prepare_data(self,df, test_split = 0.2, segment_split= False):
        print ('Split into test and trainings data...')
        
        df['touch_bin'] = df['touch'].apply(lambda x : 0 if x <= 0.5 else 1)

        if segment_split:
            segments = np.empty(len(df))
            num_segments = 0
            for idx, r in enumerate(df['reset']):
                if r == 1: num_segments = num_segments +1
                segments[idx] = num_segments
            df['segment'] = segments
            split = int(num_segments * (1- test_split))
            df_train = df[df['segment']<= split]
            df_test =  df[df['segment']> split]
        else:
            split = int(len(df) * (1- test_split))
            train = df.iloc[:split]
            test =  df.iloc[split:]

        print(train.shape, type(train))

        x_train, y_train = self.create_segments_and_labels(train, 10, 'touch_bin', 6) # is this always the same as input dim?

        num_time_periods, n_features = x_train.shape[1], x_train.shape[2]
        print(x_train.shape)

        input_shape = (num_time_periods*n_features)
        x_train = x_train.reshape(x_train.shape[0], input_shape) # is shape[0] the number of samples?
        y_train = y_train.reshape(-1,1)

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')


        return x_train, y_train#, X_test, y_test




    def create_lstm(self,input_shape, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
        print ('Create the LSTM model...')
        model = Sequential()
        model.add(tf.keras.layers.Reshape((self.TIME_PERIODS, input_dim), input_shape=(input_shape,)))
        model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (input_shape,))) # change shape to 100
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, dropout= dropout))
        model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

    def create_plots(self,history):
        print ('Plot the results...')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('./data/accuracy.png')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('./data/loss.png')
        plt.show()

        #plot_model(model, to_file='model.png')

        # validate model on unseen data
        #score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
        #print('Validation score:', score)
        print('Validation accuracy:', history.history['accuracy'])

