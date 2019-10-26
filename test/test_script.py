import sys
sys.path.append('../')

import numpy as np
import time

# Record Data
from record_cursor_data import record_data

time_max = 10
coords = record_data(time_max = time_max)

'''
FLOW:

    INITIAL_STATE_VECTOR, INITIAL_ERROR_COVARIANCE_MATRIX

    PRIORI CALCULATION, PRIORI_ERROR_COVARIANCE

    OBSERVE_READINGS, EXPECTED_READINGS, EXPECTED_ERROR_COVARIANCE

    KALMAN_GAIN
        TAKES IN MEASUREMENT_NOISE, MEASUREMENT_MATRIX, NEW_ERROR_COVARIANCE

    POSTERIORI CALCULATION
        TAKES IN KALMAN_GAIN, OBSERVED_READINGS, PRIORI

    POSTERIORI_ERROR_COVARIANCE

'''

'''

INPUT:

    cursor coords

OUTPUT:

    priori, posteriori

'''

# Dynamics
from process_dynamics import *
from kalman import Kalman

# initial_state = [x_coord, y_coord, x_velocity, y_velocity, x_acceleration, y_acceleration]
initial_state = np.asarray([coords[0][0],coords[0][1],0,0,0,0]).T

k = Kalman(initial_state = initial_state, initial_error_covariance_matrix = initial_error_covariance_matrix,
    transition_matrix = transition_matrix, measurement_matrix = measurement_matrix, input_matrix = input_matrix, input_vector = input_vector, measurement_noise = uncertainty,
    process_noise = disturbance)

# Visualization

length = 1366
breadth = 768

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

xdata, ydata = [], []
actual, = plt.plot([], [], '*', label = 'actual')

priori_x, priori_y = [], []
priori_plot, = plt.plot([], [], 'g*', label = 'priori_plot')

posteriori_x, posteriori_y = [], []
posteriori_plot, = plt.plot([], [], 'r-', label = 'posteriori_plot')

def init():
    ax.set_xlim(0, length)
    ax.set_ylim(0, breadth)
    ax.legend()
    return actual, priori_plot, posteriori_plot,

def update(frame):
    n = frame
    # print(xdata[-1], ydata[-1])
    # print(n - 1000, n)
    # print(np.asarray(coords[n - 100:n])[:, 0], 1080 - np.asarray(coords[n - 100:n])[:, 1])
    pos_x = np.asarray(coords[n])[0]
    pos_y = breadth - np.asarray(coords[n])[1]
    priori, posteriori = k(np.asarray([pos_x, pos_y]))

    # actual.set_data(pos_x, pos_y)

    priori_x.append(priori[0])
    priori_y.append(priori[1])
    priori_plot.set_data(priori_x, priori_y)

    xdata.append(pos_x)
    ydata.append(pos_y)
    actual.set_data(xdata, ydata)

    # print(priori_x[-1], priori_y[-1], xdata[-1], ydata[-1])
    # print(priori)

    posteriori_x.append(posteriori[0])
    posteriori_y.append(posteriori[1])
    posteriori_plot.set_data(posteriori_x, posteriori_y)

    return actual, priori_plot, posteriori_plot,

ani = FuncAnimation(fig, update, frames=np.arange(1, len(coords), 2),
                    init_func=init, blit=True, interval = 100)
plt.show()
