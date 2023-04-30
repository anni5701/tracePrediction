import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from data_processing import get_ith_segment
import numpy as np


def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./data/accuracy.png')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./data/loss.png')
    plt.show()


def plot_confusion_matrix(predictions, y_test):
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig('./data/confusion-matrix.png')
    plt.show()


def plot_data_set(df):
    fig, axes = plt.subplots(8, figsize=(8, 16))
    
    is_touch = df["touch"] == 1
    axes[0].plot(df["x"].where(is_touch), df["y"].where(is_touch), c="k")
    axes[0].plot(df["x"].where(~is_touch), df["y"].where(~is_touch), c="grey", alpha= 0.4)
    axes[0].set_ylabel('tablet data')

    axes[1].plot(df['arduino_timestamp'], df['ax'], c="k", label= "sensor frame")
    axes[1].plot(df['arduino_timestamp'], df['nav_ax'], c="r", label= "nav frame")
    axes[1].set_ylabel('ax  [m/s^2]')
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[2].plot(df['arduino_timestamp'], df['ay'], c="k", label= "sensor frame")
    axes[2].plot(df['arduino_timestamp'], df['nav_ay'], c="r", label= "nav frame")
    axes[2].set_ylabel('ay  [m/s^2]')
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[3].plot(df['arduino_timestamp'], df['az'], c="k", label= "sensor frame")
    axes[3].plot(df['arduino_timestamp'], df['nav_az'], c="r", label= "nav frame")
    axes[3].set_ylabel('az  [m/s^2]')
    axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axes[4].plot(df['arduino_timestamp'], df['vel_x'], c="k")
    axes[4].set_ylabel('vel x')
    axes[5].plot(df['arduino_timestamp'], df['vel_y'], c="k")
    axes[5].set_ylabel('vel y')  
    axes[6].plot(df['arduino_timestamp'], df['vel_z'], c="k")
    axes[6].set_ylabel('vel z')

    axes[7].plot(df['pos_x'], df['pos_y'], c="k")
    axes[7].set_ylabel('integrated positions')

    fig.align_ylabels()
    fig.tight_layout()

    plt.show()

def plot_ith_segment(df, i:int=0):
    fig, axes = plt.subplots(8, figsize=(8, 16))
    segment = df[df["segment"] == i]
    
    axes[0].plot(segment["x"], segment["y"], c="k")
    axes[0].set_ylabel('tablet data')

    axes[1].plot(segment['t_r'], segment['ax'], c="k", label= "sensor frame")
    axes[1].plot(segment['t_r'], segment['nav_ax'], c="r", label= "nav frame")
    axes[1].set_ylabel('ax  [m/s^2]')
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[2].plot(segment['t_r'], segment['ay'], c="k", label= "sensor frame")
    axes[2].plot(segment['t_r'], segment['nav_ay'], c="r", label= "nav frame")
    axes[2].set_ylabel('ay  [m/s^2]')
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[3].plot(segment['t_r'], segment['az'], c="k", label= "sensor frame")
    axes[3].plot(segment['t_r'], segment['nav_az'], c="r", label= "nav frame")
    axes[3].set_ylabel('az  [m/s^2]')
    axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axes[4].plot(segment['t_r'], segment['vel_x'], c="k")
    axes[4].set_ylabel('vel x')
    axes[5].plot(segment['t_r'], segment['vel_y'], c="k")
    axes[5].set_ylabel('vel y')  
    axes[6].plot(segment['t_r'], segment['vel_z'], c="k")
    axes[6].set_ylabel('vel z')

    axes[7].plot(segment['pos_x'], segment['pos_y'], c="k")
    axes[7].set_ylabel('integrated positions')

    fig.align_ylabels()
    fig.tight_layout()

    plt.show()


def plot_segment(df, index_of_segment:int= 0):

    segment = get_ith_segment(df, index_of_segment)
    fig, axes = plt.subplots(8, figsize=(8, 16))
    
    is_touch = segment["touch"] == 1
    axes[0].plot(segment["x"].where(is_touch), segment["y"].where(is_touch), c="k")
    axes[0].plot(segment["x"].where(~is_touch), segment["y"].where(~is_touch), c="grey", alpha= 0.4)
    axes[0].set_ylabel('tablet data')

    axes[1].plot(segment['arduino_timestamp'], segment['ax'], c="k", label= "sensor frame")
    axes[1].plot(segment['arduino_timestamp'], segment['nav_ax'], c="r", label= "nav frame")
    axes[1].set_ylabel('ax  [m/s^2]')
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[2].plot(segment['arduino_timestamp'], segment['ay'], c="k", label= "sensor frame")
    axes[2].plot(segment['arduino_timestamp'], segment['nav_ay'], c="r", label= "nav frame")
    axes[2].set_ylabel('ay  [m/s^2]')
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[3].plot(segment['arduino_timestamp'], segment['az'], c="k", label= "sensor frame")
    axes[3].plot(segment['arduino_timestamp'], segment['nav_az'], c="r", label= "nav frame")
    axes[3].set_ylabel('az  [m/s^2]')
    axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axes[4].plot(segment['arduino_timestamp'], segment['vel_x'], c="k")
    axes[4].set_ylabel('vel x')
    axes[5].plot(segment['arduino_timestamp'], segment['vel_y'], c="k")
    axes[5].set_ylabel('vel y')  
    axes[6].plot(segment['arduino_timestamp'], segment['vel_z'], c="k")
    axes[6].set_ylabel('vel z')

    axes[7].plot(segment['pos_x'], segment['pos_y'], c="k")
    axes[7].set_ylabel('integrated positions')

    fig.align_ylabels()
    fig.tight_layout()

    plt.show()

def plot_3d(df):
    plt.style.use('seaborn')

    fig3,ax = plt.subplots()
    fig3.suptitle('3D Trajector',fontsize=20)
    ax = plt.axes(projection='3d')
    ax.plot3D(df['pos_x'],df['pos_y'],df['pos_z'],c='red')
    ax.plot3D(df.x ,df.y, df.z, c='blue' )
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')
    plt.show()

def plot_3d_tab(df):
    plt.style.use('seaborn')

    fig3,ax = plt.subplots()
    fig3.suptitle('3D Trajector',fontsize=20)
    ax = plt.axes(projection='3d')
    ax.plot3D(df.x ,df.y, df.z, c='blue' )
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')
    plt.show()



def interactive_3d_plot_positions(df):
    fig = go.Figure(data=[go.Scatter3d(mode='markers',x= df['pos_x'], y= df['pos_y'], z =df['pos_z'], marker=dict(size=1))])
    fig.update_layout(
                title= {'text': "Integrated positions", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                scene = dict(
                    xaxis_title='X AXIS',
                    yaxis_title='Y AXIS',
                    zaxis_title='Z AXIS'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))
    fig.show()

def interactive_3d_plot_tab(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(mode='markers',x= df.x, y= df.y, z = df.z, marker=dict(size=1)))
    fig.add_trace(go.Scatter3d(mode='markers',x= [min(df.x)], y= [max(df.y)], z = [0], marker=dict(size=10), name="upper left corner tablet"))
    fig.update_layout(
                title= {'text': "Tablet data", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                scene = dict(
                    xaxis_title='X AXIS',
                    yaxis_title='Y AXIS',
                    zaxis_title='Z AXIS'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))
    fig.show()

def interactive_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(mode='markers',x= df['pos_x'], y= df['pos_y'], z = df['pos_z'], marker=dict(size=1), name="calculated positions"))
    fig.add_trace(go.Scatter3d(mode='markers',x= df.x, y= df.y, z = df.z, marker=dict(size=1), name="tablet data"))
    fig.add_trace(go.Scatter3d(mode='markers',x= [min(df.x)], y= [max(df.y)], z = [0], marker=dict(size=10), name="upper left corner tablet"))
    fig.update_layout(
                title= {'text': "Combined", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                scene = dict(
                    xaxis_title='X AXIS',
                    yaxis_title='Y AXIS',
                    zaxis_title='Z AXIS'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))
    
    fig.show()


def test_predcition_comparison(y_test, predictions):
    x_true = y_test[:,0]
    y_true = y_test[:,1]
    x_pred = predictions[:,0]
    y_pred = predictions[:,1]

    print("First predictions for x: ", x_pred[:5])
    print("First predictions for y: ",y_pred[:5])

    plt.plot(x_true,y_true, c='blue', label='true values')
    plt.plot(x_pred, y_pred, c='red', label='predicted values')
    plt.legend()
    plt.show()


def test_predcition_comparison_x(y_test, predictions):
    x_true = y_test[:,0]
    x_pred = predictions[:,0]
    t = range(0,len(x_true))

    plt.plot(t, x_true, c='blue', label='true values')
    plt.plot(t, x_pred, c='red', label='predicted values')
    plt.legend()
    plt.show()


def test(y_test, predictions):
    pos_values = np.array([y_test[0,0], y_test[0,1]]) # choose y_test start points
    for i in range(1, len(predictions)-1):
        np.vstack(pos_values, np.array([pos_values[i-1,0], pos_values[i-1,1]]))

    plt.plot(y_test[:,0], y_test[:,1], c='blue', label='true values')
    plt.plot(pos_values[:,0], pos_values[:,1], c='red', label='predicted values')
    plt.legend()
    plt.show()
    
