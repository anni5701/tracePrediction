import plotly.graph_objects as go
import matplotlib.pyplot as plt


def plot_segments(df):
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

def plot_3d(self, df):
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

def plot_3d_tab(self, df):
    plt.style.use('seaborn')

    fig3,ax = plt.subplots()
    fig3.suptitle('3D Trajector',fontsize=20)
    ax = plt.axes(projection='3d')
    ax.plot3D(df.x ,df.y, df.z, c='blue' )
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')
    plt.show()



def interactive_3d_plot_positions(self, df):
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

def interactive_3d_plot_tab(self, df):
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

def interactive_plot(self, df):
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
