import numpy as np

initial_error_covariance_matrix = 0*np.eye(6)

dt = 0.02

transition_matrix = np.asarray([[1,0,dt,0,0,0],
                                [0,1,0,dt,0,0],
                                [0,0,1,0,dt,0],
                                [0,0,0,1,0,dt],
                                [0,0,0,0,1,0],
                                [0,0,0,0,0,1]])

input_matrix = np.zeros(6).T
input_vector = np.zeros((1,1))

measurement_matrix = np.asarray([[1,0,0,0,0,0],
                                 [0,1,0,0,0,0]])

disturbance = 10*np.eye(6)

uncertainty = 20*np.eye(2)
