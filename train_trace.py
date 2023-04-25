import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from visualization import *
from sklearn.preprocessing import StandardScaler

# only use segements
# use segments to split
# removed comma from split indices
# introduce empty list of split indices before assigning value with where so that I can also use it on df without segments



class PredictionModelTrace():

    BATCH_SIZE = 52 
    INPUT_DIM = 16
    OUTPUT_DIM = 3 
    RNN_HIDDEN_DIM = 128

    PAD_VALUE = -1000

    EPCOHS = 50
    TIME_PERIODS = 5

    LENGTH_SEGMENT = None

    def __init__(self, dataframe, segments: bool= False):

        self.segments = segments
        if segments:
            x_train, y_train, x_test, y_test = self.prepare_data_with_segments(dataframe)  
            model = self.create_lstm1(x_train.shape[1], x_train.shape[2]) 
        else: 
            x_train, y_train, x_test, y_test = self.prepare_data(dataframe)  
            model = self.create_lstm(input_shape=x_train.shape[1])  

        print("The trainings data is of shape {} and {} and the test data is of shape {} and {}". format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        print ('Fitting model...')
        
        history = model.fit(x_train, y_train , epochs=self.EPCOHS,  validation_split = 0.2, verbose = "2") #, shuffle= "batch", true batch_size= self.BATCH_SIZE ,

        plot_loss(history)

        predictions = self.predictions_test_data(model, x_test, y_test)
        test_predcition_comparison(y_test, predictions)
        test_predcition_comparison_x(y_test, predictions)


        

    def predictions_test_data(self, model, x_test, y_test):
        print ('Use  model on test data...')
        predictions = model.predict(x_test)
        if not self.segments:
            print("The loss on the test data is {}".format(mean_squared_error(y_test, predictions)))
        return predictions
    # warum hier nicht mean für die verschiedenen timestemps
        

    def create_segments_and_labels(self, dataframe: pd.DataFrame , step: int, n_features: int):
        print ('Create segments and labels...')
        segments = []
        labels = []
        for i in range(0, len(dataframe) - step, step):
            vel_x = dataframe['vel_x'].values[i: i + step]
            vel_y = dataframe['vel_y'].values[i: i + step]
            vel_z = dataframe['vel_z'].values[i: i + step]
            nav_x = dataframe['nav_ax'].values[i: i + step]
            nav_y = dataframe['nav_ay'].values[i: i + step]
            nav_z = dataframe['nav_az'].values[i: i + step]
            ax = dataframe['ax'].values[i: i + step]
            ay = dataframe['ay'].values[i: i + step]
            az = dataframe['az'].values[i: i + step]
            gx = dataframe['gx'].values[i: i + step]
            gy = dataframe['gy'].values[i: i + step]
            gz = dataframe['gz'].values[i: i + step]
            q0 = dataframe['q0'].values[i: i + step]
            q1 = dataframe['q1'].values[i: i + step]
            q2 = dataframe['q2'].values[i: i + step]
            q3 = dataframe['q3'].values[i: i + step]

            # Retrieve the mean of the labels in this segment
            x,y,z = dataframe.iloc[i: i + step, 9:12].mean(axis=0) #['x', 'y', 'z']
            segments.append([vel_x, vel_y, vel_z, nav_x, nav_y, nav_z, ax, ay, az, gx, gy, gz, q0, q1, q2, q3])
            labels.append([x,y,z])

        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, step, n_features)
        labels = np.asarray(labels, dtype= np.float32)

        return reshaped_segments, labels
    
    def extract_segments_and_labels(self, dataframe: pd.DataFrame):
        print ('Extract segments and labels...')
        segments = []
        labels = []
        split_indices = np.where(dataframe["reset"] == 1) 
        # add beginning and end to the split indices
        split_indices = np.insert(split_indices, 0, 0)
        split_indices = np.append(split_indices, len(dataframe)-1)

        if self.LENGTH_SEGMENT == None:
            self.LENGTH_SEGMENT = max(np.diff(split_indices))

        for i in range(0,len(split_indices)-1):
            vel_x = dataframe['vel_x'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            vel_y = dataframe['vel_y'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            vel_z = dataframe['vel_z'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            nav_x = dataframe['nav_ax'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            nav_y = dataframe['nav_ay'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            nav_z = dataframe['nav_az'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            ax = dataframe['ax'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            ay = dataframe['ay'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            az = dataframe['az'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            gx = dataframe['gx'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            gy = dataframe['gy'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            gz = dataframe['gz'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            q0 = dataframe['q0'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            q1 = dataframe['q1'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            q2 = dataframe['q2'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            q3 = dataframe['q3'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)

            x = dataframe['x'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            y = dataframe['y'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)
            z = dataframe['z'].values[split_indices[i]: split_indices[i+1]].reshape(-1,1)

            features = np.hstack((vel_x, vel_y, vel_z, nav_x, nav_y, nav_z, ax, ay, az, gx, gy, gz, q0, q1, q2, q3)).flatten()
            positions = np.hstack((x,y,z)).flatten()


            segments.append(features)
            labels.append(positions)

        segments = tf.keras.preprocessing.sequence.pad_sequences(segments, padding="post", value=self.PAD_VALUE, maxlen= self.LENGTH_SEGMENT * self.INPUT_DIM)
        labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding="post", value=self.PAD_VALUE, maxlen= self.LENGTH_SEGMENT * self.OUTPUT_DIM)

        print(np.asarray(segments, dtype= np.float32).shape)
        print(np.asarray(labels, dtype= np.float32).shape)

        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(len(split_indices)-1,-1, self.INPUT_DIM)
        reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(len(split_indices)-1,-1, self.OUTPUT_DIM)

        print("reshaped labels shape: ",reshaped_labels.shape, "reshaped segments shape: ", reshaped_segments.shape)

        print(reshaped_labels[:1,:10,:])

        return reshaped_segments, reshaped_labels

    def prepare_data(self,df, test_split = 0.2,):
        print ('Split into test and trainings data...')

        #scale features with min-max-scaler
        features = ['ax', 'ay', 'az', 'gx', 'gy','gz','q0', 'q1', 'q2', 'q3', 'nav_ax', 'nav_ay', 'nav_az','vel_x', 'vel_y', 'vel_z']
        scaler = StandardScaler() 
        df[features] = pd.DataFrame(scaler.fit_transform(df[features].values), index=df.index)

        #split into training and test data
        split = int(len(df) * (1- test_split))
        train_data = df.iloc[:split]
        test_data =  df.iloc[split:]

        x_train, y_train = self.create_segments_and_labels(train_data, self.TIME_PERIODS, self.INPUT_DIM)
        x_test, y_test = self.create_segments_and_labels(test_data, self.TIME_PERIODS, self.INPUT_DIM)

        num_time_periods, n_features = x_train.shape[1], x_train.shape[2]

        input_shape = (num_time_periods*n_features) # num segments times num features
        x_train = x_train.reshape(x_train.shape[0], input_shape) 
        y_train = y_train.reshape(-1, self.OUTPUT_DIM)

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        num_time_periods, n_features = x_test.shape[1], x_test.shape[2]
        input_shape = (num_time_periods*n_features)
        x_test = x_test.reshape(x_test.shape[0], input_shape)
        y_test = y_test.reshape(-1, self.OUTPUT_DIM)

        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')


        return x_train, y_train, x_test, y_test
    
    def prepare_data_with_segments(self,df, test_split = 0.2,):
        print ('Split into test and trainings data...')

        #scale features with min-max-scaler
        features = ['ax', 'ay', 'az', 'gx', 'gy','gz','q0', 'q1', 'q2', 'q3', 'nav_ax', 'nav_ay', 'nav_az','vel_x', 'vel_y', 'vel_z']
        scaler = StandardScaler() 
        df[features] = pd.DataFrame(scaler.fit_transform(df[features].values), index=df.index)

        #split into training and test data
        split = int(len(df) * (1- test_split))
        train_data = df.iloc[:split]
        test_data =  df.iloc[split:]

        x_train, y_train = self.extract_segments_and_labels(train_data)
        x_test, y_test = self.extract_segments_and_labels(test_data)


        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')


        return x_train, y_train, x_test, y_test




    def create_lstm(self,input_shape, input_dim = INPUT_DIM):
        print ('Create the LSTM model...')
        model = Sequential()
        model.add(tf.keras.layers.Reshape((self.TIME_PERIODS, input_dim), input_shape=(input_shape,)))
        model.add(tf.keras.layers.LSTM(units = self.RNN_HIDDEN_DIM, input_shape = (input_shape,))) # return_sequences = True,
        #model.add(tf.keras.layers.LSTM(units = 10, return_sequences = True))
        #model.add(tf.keras.layers.Dense(units = self.OUTPUT_DIM, activation='relu'))
        #model.add(tf.keras.layers.LSTM(units = 10, return_sequences = True)) 
        #model.add(tf.keras.layers.LSTM(units = self.RNN_HIDDEN_DIM)) 
        model.add(tf.keras.layers.Dense(units = self.OUTPUT_DIM)) 
        model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        model.compile('adam', 'mean_squared_error', 'mean_squared_error') #optimizer, loss, metrics
        print(model.summary())
        return model

    def create_lstm1(self,n_timesteps, n_features):
        print ('Create the LSTM model...')
        model = Sequential()
        model.add(tf.keras.layers.Masking(mask_value=self.PAD_VALUE, input_shape = (n_timesteps,n_features)))
        model.add(tf.keras.layers.LSTM(units = self.OUTPUT_DIM, time_major=False, return_sequences=True)) # return_sequences = True,
        model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        model.compile('adam', 'mean_squared_error', 'mean_squared_error') #optimizer, loss, metrics
        print(model.summary())
        return model


