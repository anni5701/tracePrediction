import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from visualization import *
from sklearn.preprocessing import StandardScaler
import itertools


# only use segements
# use segments to split
# removed comma from split indices
# introduce empty list of split indices before assigning value with where so that I can also use it on df without segments



class PredictionModelTrace():

    BATCH_SIZE = 200 
    INPUT_DIM = 16
    OUTPUT_DIM = 1 # check for x first
    RNN_HIDDEN_DIM = 128

    PAD_VALUE = 999

    EPOCHS = 10

    LENGTH_SEGMENT = None

    def __init__(self, dataframe):

        self.num_segments = dataframe["segment"].iloc[-1]

        x_train, y_train, x_test, y_test = self.prepare_data_with_segments(dataframe)
        #x_train_mask = tf.not_equal(x_train, 0)
        #x_test_mask = tf.not_equal(x_test, 0)
  
       # model = self.create_lstm1(x_train.shape[1], x_train.shape[2]) 
     

        print("The trainings data is of shape {} and {} and the test data is of shape {} and {}". format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        print ('Fitting model...')

        model = self.get_model_deep(x_train.shape[1], self.INPUT_DIM)

        for i, l in enumerate(model.layers):
            print(f'layer {i}: {l}')
            print(f'\thas input mask: {l.input_mask}')
            print(f'\thas output mask: {l.output_mask}')

        model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.001), loss='mae')
        print(model.summary())
        

        history = model.fit(x_train, y_train, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, validation_split=0.2)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        print ('Use  model on test data...')
        y_pred = model.predict(x_test, batch_size=self.BATCH_SIZE)

        print(y_pred.shape)
        print(y_pred[:10])

        #plt.plot(range(self.LENGTH_SEGMENT), y_pred[0,:,:], c="red", label="predictions")
        #plt.plot(range(self.LENGTH_SEGMENT), y_test[0,:,:], c="blue", label="true values")
        #plt.ylabel('x values')
        #plt.legend()
        #plt.show()

        
        
        #history = model.fit(x_train, y_train , epochs=self.EPOCHS,  validation_split = 0.2, verbose = "2") #, shuffle= "batch", true batch_size= self.BATCH_SIZE ,

        #plot_loss(history)

        #score = model.evaluate(x_test, y_test, verbose="0")
        #print('Test loss:', score[0])
        #print('Test accuracy:', score[1])

        #predictions = self.predictions_test_data(model, x_test, y_test)
        #test_predcition_comparison(y_test, predictions)
        #test_predcition_comparison_x(y_test, predictions)


        

    def predictions_test_data(self, model, x_test, y_test):
        print ('Use  model on test data...')
        predictions = model.predict(x_test)
        return predictions
    # warum hier nicht mean für die verschiedenen timestemps

    def extract_segments_and_labels(self, dataframe: pd.DataFrame, y:bool= True):
        print ('Extract segments and labels...')
        segments = []
        labels = []

        if self.LENGTH_SEGMENT == None:
            self.LENGTH_SEGMENT = np.max(dataframe.groupby("segment").size())

        for i in range(self.num_segments):
            d_i = dataframe[dataframe["segment"] == i].iloc[:,1:] # do not use t column as feature
            d_i = d_i.drop("segment", axis=1)
            d_i = d_i.drop("r", axis=1)
            #ax,ay,az, gx,gy,gz, q0,q1,q2,q3,r,nav_ax,nav_ay,nav_az,vel_x,vel_y,vel_z,pos_x,pos_y,pos_z
            features = d_i.iloc[:,3:].to_numpy().flatten().tolist()
            positions = d_i.iloc[:,:1].to_numpy().flatten().tolist() #xyz #HIER geändert zu nur x erstmal

            segments.append(features)
            labels.append(positions)

        segments = tf.keras.preprocessing.sequence.pad_sequences(segments, padding="post", value=self.PAD_VALUE, maxlen= self.LENGTH_SEGMENT * self.INPUT_DIM)
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(self.num_segments,-1, self.INPUT_DIM)

        print(reshaped_segments[0,-100:0,:])

        if y:
            labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding="post", value=self.PAD_VALUE, maxlen= self.LENGTH_SEGMENT * self.OUTPUT_DIM)
            reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(self.num_segments,-1, self.OUTPUT_DIM)
        else:
            labels = list(itertools.chain(*labels))
            reshaped_labels = np.asarray(labels, dtype= np.float32)

        print("reshaped labels shape: ",reshaped_labels.shape, "reshaped segments shape: ", reshaped_segments.shape)
        #print(reshaped_labels[:1,:10,:])

        return reshaped_segments, reshaped_labels


    
    def prepare_data_with_segments(self,df, test_split = 0.2,):
        print ('Split into test and trainings data...')

        #scale features 
        features = ['ax', 'ay', 'az', 'gx', 'gy','gz','q0', 'q1', 'q2', 'q3', 'nav_ax', 'nav_ay', 'nav_az','vel_x', 'vel_y', 'vel_z']
        scaler = StandardScaler() 
        df[features] = pd.DataFrame(scaler.fit_transform(df[features].values), index=df.index)

        #split into training and test data
        split = int(self.num_segments * (1- test_split))
        train_data = df[df["segment"] <= split]
        test_data =  df[df["segment"] > split]

        x_train, y_train = self.extract_segments_and_labels(train_data)
        x_test, y_test = self.extract_segments_and_labels(test_data)  #, y=False

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')

        return x_train, y_train, x_test, y_test



    def create_lstm1(self,n_timesteps, n_features):
        print ('Create the LSTM model...')
        model = Sequential()
        model.add(tf.keras.layers.Masking(mask_value=self.PAD_VALUE, input_shape = (n_timesteps,n_features)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        model.add(tf.keras.layers.LSTM(units = self.OUTPUT_DIM, time_major=False, return_sequences=True)) 
        model.compile('adam', 'mean_squared_error', 'mean_squared_error') #optimizer, loss, metrics
        print(model.summary())
        return model


    

    def get_model_deep(self, max_seg_length, num_features=INPUT_DIM):
        inp = tf.keras.layers.Input((max_seg_length,num_features))
        x = tf.keras.layers.Masking(mask_value=self.PAD_VALUE)(inp)
        x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))(x) #TimeDistributed(
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.OUTPUT_DIM))(x)

        model = tf.keras.Model(inp, x)
        tf.keras.Model()
        return model
    

