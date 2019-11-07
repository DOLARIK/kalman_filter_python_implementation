def test(noise = 20, time_max = 10, show_acceleration_plots = True, filter_name = 'kalman'):

    import sys
    sys.path.append('../')

    import numpy as np
    import time

    # Record Data
    import time
    import numpy as np
    import pyautogui

    def record_data(noise = 20, time_max = 10):

        print("RECORDING DATA")

        print("Move your cursor as if you were driving it.")

        coords = []
        coords_actual = []
        timestamp = []

        T = time.time()

        while time.time() - T < time_max:

            t = time.time()
            while True:
                if time.time() - t > .02:

                    timestamp.append(time.time() - T)

                    act = np.asarray(pyautogui.position())

                    pos = act + 2*noise*np.asarray([np.random.rand(), np.random.rand()]) - noise*np.asarray([np.random.rand(), np.random.rand()])
                    coords.append(pos)

                    pos_actual = act
                    coords_actual.append(pos_actual)

                    break

        print("RECORDING DATA COMPLETED")

        return  coords, coords_actual, timestamp

    coords, coords_actual, timestamp = record_data(noise = noise, time_max = time_max)

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

    # initial_state = [x_coord, y_coord, x_velocity, y_velocity, x_acceleration, y_acceleration]
    initial_state = np.asarray([coords[0][0],coords[0][1],0,0,0,0]).T

    initial_error_covariance_matrix = 0*np.eye(6)

    dt = 0.02

    transition_matrix = np.asarray([[1,0,dt,0,0,0],
                                    [0,1,0,dt,0,0],
                                    [0,0,1,0,dt,0],
                                    [0,0,0,1,0,dt],
                                    [0,0,0,0,1,0],
                                    [0,0,0,0,0,1]])

    position_funciton = lambda position, velocity : position + velocity*dt
    velocity_funciton = lambda velocity, acceleration : velocity + acceleration*dt
    acceleration_function = lambda acceleration : acceleration

    def transition_function(state_vector, input_vector):

        # print('state_vector', state_vector.shape)

        position = position_funciton(state_vector[0:2], state_vector[2:4])
        velocity = velocity_funciton(state_vector[2:4], state_vector[4:6])
        acceleration = acceleration_function(state_vector[4:6])

        return_state = np.vstack((position, velocity, acceleration))

        return np.reshape(return_state, (-1, 1)) + np.zeros(6).T.reshape(state_vector.size, input_vector.shape[0])*input_vector

    def measurement_function(state_vector):
        # print('state_vector', state_vector.shape)
        return np.reshape(state_vector[0:2], (-1,1))

    # print(nd.Jacobian(transition_function)(initial_state))

    input_matrix = np.zeros(6).T
    input_vector = np.zeros((1,1))

    measurement_matrix = np.asarray([[1,0,0,0,0,0],
                                     [0,1,0,0,0,0]])

    disturbance = 10*np.eye(6)

    uncertainty = noise*np.eye(2)

    from kalman import Kalman
    from extended_kalman import ExtendedKalman

    kf = Kalman(initial_state = initial_state, initial_error_covariance_matrix = initial_error_covariance_matrix,
        transition_matrix = transition_matrix, measurement_matrix = measurement_matrix, input_matrix = input_matrix, input_vector = input_vector, measurement_noise = uncertainty,
        process_noise = disturbance)

    ekf = ExtendedKalman(initial_state = initial_state, initial_error_covariance_matrix = initial_error_covariance_matrix,
        transition_function = transition_function, measurement_function = measurement_function, input_matrix = input_matrix, input_vector = input_vector, measurement_noise = uncertainty,
        process_noise = disturbance)

    if filter_name == 'kalman':
        k = kf
    elif filter_name == 'extended':
        k = ekf

    # Visualization

    length = 1920
    breadth = 1080

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()

    cursor_x, cursor_y = [], []
    actual, = plt.plot([], [], '-', label = 'actual')

    xdata, ydata = [], []
    recorded, = plt.plot([], [], '*', label = 'recorded')

    priori_x, priori_y = [], []
    priori_plot, = plt.plot([], [], 'g-', label = 'priori_plot')

    posteriori_x, posteriori_y = [], []
    posteriori_plot, = plt.plot([], [], 'r-', label = 'posteriori_plot')

    if show_acceleration_plots:

        fig_acc_x, acc_x = plt.subplots()
        priori_acceleration_x = []
        posteriori_acceleration_x = []
        priori_acceleration_plot_x, = plt.plot([], [], 'g-', label = 'priori_acc_x')
        posteriori_acceleration_plot_x, = plt.plot([], [], 'r-', label = 'posteriori_acc_x')

        fig_acc_y, acc_y = plt.subplots()
        priori_acceleration_y = []
        posteriori_acceleration_y = []
        priori_acceleration_plot_y, = plt.plot([], [],'g-', label = 'priori_acc_y')
        posteriori_acceleration_plot_y, = plt.plot([], [], 'r-', label = 'posteriori_acc_y')

        timestamp_data = []


    def init():
        ax.set_xlim(0, length)
        ax.set_ylim(0, breadth)
        ax.legend()

        if show_acceleration_plots:

            acc_x.set_xlim(0, 10)
            acc_x.set_ylim(-1000, 1000)
            acc_x.legend()

            acc_y.set_xlim(0, 10)
            acc_y.set_ylim(-1000, 1000)
            acc_y.legend()

            return actual, recorded, priori_plot, posteriori_plot, priori_acceleration_plot_x, posteriori_acceleration_plot_x, priori_acceleration_plot_y, posteriori_acceleration_plot_y,

        else:
            return actual, recorded, priori_plot, posteriori_plot,



    def update(frame):
        n = frame
        # print(xdata[-1], ydata[-1])
        # print(n - 1000, n)
        # print(np.asarray(coords[n - 100:n])[:, 0], 1080 - np.asarray(coords[n - 100:n])[:, 1])
        pos_x = np.asarray(coords[n])[0]
        pos_y = breadth - np.asarray(coords[n])[1]

        # kpriori, kposteriori = kf(np.asarray([pos_x, pos_y]), None)
        # epriori, eposteriori = ekf(np.asarray([pos_x, pos_y]), None)

        priori, posteriori = k(np.asarray([pos_x, pos_y]), input_vector)

        priori_x.append(priori[0])
        priori_y.append(priori[1])
        priori_plot.set_data(priori_x, priori_y)

        xdata.append(pos_x)
        ydata.append(pos_y)
        recorded.set_data(xdata, ydata)

        posteriori_x.append(posteriori[0])
        posteriori_y.append(posteriori[1])
        posteriori_plot.set_data(posteriori_x, posteriori_y)

        cursor_x.append(np.asarray(coords_actual[n])[0])
        cursor_y.append(breadth - np.asarray(coords_actual[n])[1])
        actual.set_data(cursor_x, cursor_y)


        if show_acceleration_plots:

            timestamp_data.append(timestamp[n])

            priori_acceleration_x.append(priori[4,0])
            priori_acceleration_plot_x.set_data(timestamp_data, priori_acceleration_x)

            posteriori_acceleration_x.append(posteriori[4,0])
            posteriori_acceleration_plot_x.set_data(timestamp_data, posteriori_acceleration_x)

            priori_acceleration_y.append(priori[5,0])
            priori_acceleration_plot_y.set_data(timestamp_data, priori_acceleration_y)

            posteriori_acceleration_y.append(posteriori[5,0])
            posteriori_acceleration_plot_y.set_data(timestamp_data, posteriori_acceleration_y)

            return actual, recorded, priori_plot, posteriori_plot, priori_acceleration_plot_x, posteriori_acceleration_plot_x, priori_acceleration_plot_y, posteriori_acceleration_plot_y,

        else:
            return actual, recorded, priori_plot, posteriori_plot,


    ani = FuncAnimation(fig, update, frames=np.arange(1, len(coords), 2),
                        init_func=init, blit=True, interval = 100)
    plt.show()



if __name__ == "__main__":
    test()
