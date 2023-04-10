import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ahrs
from numpy.linalg import norm


class Dataset:

    ACC_OFFSET = [0.00456531,  0.00791233, -0.02643263]
    ACC_SCALE = [0.10216491, 0.10159286, 0.10136561] 
    GYRO_OFFSET = [-0.019417  ,  0.05761032,  0.03028953]
    PATH_TO_DATA = "./data/"

    # use Madgwick filter to compute quaternion representation
    madgwick = ahrs.filters.Madgwick()

    def __init__(self, imu_location:str, tab_location:str, segments: bool=False):
        """
        Input: 
            imu: csv file name with imu data in form of [host_timestamp, arduino_timestamp, ax, ay, az, gx, gy, gz, temperature]
            tab: csv file name with tablet data in form of [host_timestamp, x, y, z, in_range, touch, pressure, reset]

        Output: 
            pd.DataFrame or Dictonary with segments as keys (e.g. segment_1) containing the processed and transformed data
        """

        imu = pd.read_csv(self.PATH_TO_DATA + imu_location, encoding="utf-8")
        tab = pd.read_csv(self.PATH_TO_DATA + tab_location, encoding="utf-16")

        self.data_set = self.load_data(imu,tab, segments)
        

    def load_data(self, imu: pd.DataFrame, tab:pd.DataFrame, segments:bool= False):
        print(type(imu))
        imu[['gx', 'gy', 'gz']] = imu[['gx', 'gy', 'gz']] + self.GYRO_OFFSET
        imu[['ax', 'ay', 'az']] = imu[['ax', 'ay', 'az']] * self.ACC_SCALE  + self.ACC_OFFSET

        df = self.merge_data(imu,tab)
        #df = self.adjust_timestamps(df)
        if segments:
            d = self.split_into_segments(df)
            for segment in d.keys():
                print("processing {}".format(segment))
                d[segment] = self.process_dataframe(d[segment])
            return d
        else:
            df = self.process_dataframe(df)
            return df
        
        

    def process_dataframe(self, df: pd.DataFrame):
        df = self.quaternions(df)
        print("calculate the rotation matrices...")
        df['r'] = df.apply(lambda row: self.rotation_matrix([row.q0, row.q1, row.q2, row.q3]), axis=1)
        print("calculate the navigation frame representation...")
        df[['nav_ax', 'nav_ay', 'nav_az']] = df.apply(lambda row: pd.Series(self.navigation_acc(row.r, [row.ax, row.ay, row.az])), axis=1) 

        print("integrate the acceleration...")
        vel_x, vel_y,vel_z,pos_x, pos_y, pos_z = self.integrate(df)
        integrated_data = np.stack((vel_x, vel_y, vel_z, pos_x, pos_y, pos_z), axis=1)
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(integrated_data, columns=['vel_x','vel_y','vel_z', 'pos_x', 'pos_y', 'pos_z'])], axis=1)
        return df


    def merge_data(self, imu, tab):
        """
        Input: 
            imu: pd.DataFrame with columns [host_timestamp, arduino_timestamp, ax, ay, az, gx, gy, gz, temperature]
            tab: pd.DataFrame with columns [host_timestamp, x, y, z, in_range, touch, pressure, reset]

        Output: 
            pd.DataFrame containing imu and tab data with interpolated values for the tab data
        """

        print("merging the data sets...")

        t_left, t_right = tab["host_timestamp"].iloc[[0, -1]]
        i_left, i_right = imu["host_timestamp"].iloc[[0, -1]]
        left = max(t_left, i_left)
        right = min(t_right, i_right)

        # use imu data set as base to merge on
        df = imu[(imu["host_timestamp"] >= left) & (imu["host_timestamp"] <= right)].copy() 

        for column in ["x", "y", "z", "in_range", "touch", "pressure"]:
            df[column] = np.interp(df["host_timestamp"], tab["host_timestamp"], tab[column])

        # For reset, assign to closest timestamp
        mask = tab["reset"] != 0
        indices = np.searchsorted(df["host_timestamp"], tab.loc[mask, "host_timestamp"])
        df["reset"] = 0
        df.loc[df.index[indices], "reset"] = tab.loc[mask, "reset"].values

        return df



    def adjust_timestamps(self, df: pd.DataFrame):
        print("adjust the timestamps...")

        df['arduino_timestamp'] = df['arduino_timestamp'].apply(lambda x: (x - min(df['arduino_timestamp'])) /1000)
        return df

    # split data into segments (individual sentences)
    def split_into_segments(self, df: pd.DataFrame):
        data_segments = {}
        split_indices, = np.where(df["reset"] == 1)
        if split_indices == []:
            return df # no segments reported
        for i in range(len(split_indices)-1):
            data_segments["segment_"+str(i)] = df.iloc[split_indices[i]:split_indices[i+1]]
        return data_segments


    def quaternions(self, df: pd.DataFrame): # input dataframe/single segment
        print("calculate the quaternion representations...")

        T = df["arduino_timestamp"].values
        A = df[["ax", "ay", "az"]].values
        G = df[["gx", "gy", "gz"]].values
        Q = np.zeros((len(T), 4))
        Q[0] = ahrs.common.orientation.acc2q(A[0]) # this probably assumes that az is the gravity axis
        #Q[0] = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(1, len(T)):
            self.madgwick.Dt = (T[i] - T[i - 1]) 
            Q[i] = self.madgwick.updateIMU(Q[i - 1], G[i], A[i])
        df_q = pd.concat([df.reset_index(drop=True), pd.DataFrame(Q, columns=['q0','q1','q2','q3'])], axis=1)

        return df_q

    # compute rotation matrix (only for one array of quaternion)
    def rotation_matrix(self, q):
        """
        Input:
            Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

        Output:
            3x3 rotation matrix. 
            This rotation matrix converts a point in the local reference 
            frame to a point in the navigation reference frame.
        """

        # divide by norm so we have a unit quaternion
        q0 = q[0] / norm(q)
        q1 = q[1] / norm(q)
        q2 = q[2] / norm(q)
        q3 = q[3] / norm(q)

        R = np.zeros((3,3))
        
        R[0,0] = 1- 2 * (q2 * q2 + q3 * q3) 
        R[0,1] = 2 * (q1 * q2 - q0 * q3)
        R[0,2] = 2 * (q1 * q3 + q0 * q2)
        
        R[1,0] = 2 * (q1 * q2 + q0 * q3)
        R[1,1] = 1 - 2 * (q1 * q1 + q3 * q3) 
        R[1,2] = 2 * (q2 * q3 - q0 * q1)
        
        R[2,0] = 2 * (q1 * q3 - q0 * q2)
        R[2,1] = 2 * (q2 * q3 + q0 * q1)
        R[2,2] = 1 - 2 * (q1 * q1 + q2 * q2) 

        # inverse of R is body to global, matrix is orthogonal so transpose is equal to inverse                  
        return np.transpose(R)


    def navigation_acc(self, R: np.ndarray, acc: list):
        """
        Input:
            R: IMU body frame to global frame rotation matrix
            acc: acceleration in the body frame
        Output: 
            array of acceleration in the global frame
        """
        nav_acc = R @ acc
        return nav_acc[0], nav_acc[1], nav_acc[2] # shape (3,)



    def integrate_1d(self, t, dx, x0=0):
        (n,) = dx.shape
        x = np.zeros_like(dx)
        x[0] = x0
        for i in range(1, n):
            dt = (t[i] - t[i - 1]) 
            x[i] = (dx[i - 1] + dx[i]) / 2 * dt + x[i - 1]
            
        return x

    def integrate(self, df):
        T = df["arduino_timestamp"].values
        vel_x = self.integrate_1d(T,df['nav_ax'])
        vel_y = self.integrate_1d(T,df['nav_ay'])
        vel_z = self.integrate_1d(T,df['nav_az'])

    # use starting point tab as intial integration start
        pos_x = self.integrate_1d(T,vel_x, df.x[0])
        pos_y = self.integrate_1d(T,vel_y, df.y[0])
        pos_z = self.integrate_1d(T,vel_z, df.z[0])

        return vel_x, vel_y,vel_z,pos_x,pos_y,pos_z


