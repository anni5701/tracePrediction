import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import statistics


class PredictionModel:

    BATCH_SIZE = 100 
    INPUT_DIM = 6
    OUTPUT_DIM = 1 
    RNN_HIDDEN_DIM = 32

    EPCOHS = 50
    TIME_PERIODS = 10

    def __init__(self, dataframe):
        x_train, y_train, x_test, y_test = self.prepare_data(dataframe)   
        print("The trainings data is of shape {} and the test data is of shape {}". format(x_train.shape, x_test.shape))
        neg_train, pos_train = np.bincount(y_train[:,0].astype(int))
        neg_test, pos_test = np.bincount(y_test[:,0].astype(int))
        print('Train data:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(len(y_train), pos_train, 100 * pos_train / len(y_train)))
        print('Test data:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(len(y_test), pos_test, 100 * pos_test / len(y_test)))

        initial_bias = np.log([(pos_train + pos_test) / (neg_train + neg_test)])

        model = self.create_lstm(input_shape=x_train.shape[1]) # , output_bias=initial_bias

        print ('Fitting model...')
        
        history = model.fit(x_train, y_train ,batch_size= self.BATCH_SIZE , epochs=self.EPCOHS,  validation_split = 0.2, verbose = "2") #, shuffle= "batch"

        print ('Use  mmodel on test data...')
        predictions = model.predict(x_test)
        predictions = np.mean(predictions, axis=1)
        predictions = np.where(predictions <= 0.5, 0, 1) 
        self.create_plots(history, predictions, y_test)

        
        


        

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

        x_train, y_train = self.create_segments_and_labels(train, 10, 'touch_bin', 6) # is this always the same as input dim?
        x_test, y_test = self.create_segments_and_labels(test, 10, 'touch_bin', 6) # is this always the same as input dim?

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
        model.add(tf.keras.layers.LSTM(units = 10, input_shape = (input_shape,))) # return_sequences = True,
        #model.add(tf.keras.layers.LSTM(units = 10, return_sequences = True))
        model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid', bias_initializer=output_bias)) 
        model.compile('adam', 'binary_crossentropy', 'accuracy') #metrics=[tf.keras.metrics.BinaryAccuracy()]
        print(model.summary())
        return model

    def create_plots(self,history, predictions, y_test):
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

        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()

        print('Validation accuracy:', history.history['accuracy'])

