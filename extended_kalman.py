from kalman import Kalman
import numpy as np
import numdifftools as nd

class ExtendedKalman(Kalman):

    def __init__(self, initial_state, initial_error_covariance_matrix,
                        transition_function, process_noise,
                        input_matrix, input_vector,
                        measurement_function, measurement_noise):

        self.transition_function = transition_function
        self.measurement_function = measurement_function

        super().__init__(
            initial_state = initial_state,
            initial_error_covariance_matrix = initial_error_covariance_matrix,
            transition_matrix = self.compute_jacobian(transition_function, initial_state, input_vector),
            process_noise = process_noise,
            input_matrix = input_matrix,
            input_vector = input_vector,
            measurement_matrix = self.compute_jacobian(measurement_function, initial_state),
            measurement_noise = measurement_noise)

    def __call__(self, sensor_readings, input_vector):

        # PRIORI, POSTERIORI CALCULATION
        priori, posteriori = super().__call__(sensor_readings, input_vector)

        # JACOBIAN TRANSITION AND MEASUREMENT MATRIX COMPUTATION AND UPDATION
        self.transition_matrix = self.compute_jacobian(self.transition_function, posteriori, input_vector)
        self.measurement_matrix = self.compute_jacobian(self.measurement_function, posteriori)

        # print(self.transition_matrix, self.measurement_matrix)

        return priori, posteriori

    def priori_calculation(self, state_vector, input_vector):
        return self.transition_function(state_vector, input_vector)

    def expected_readings_calculation(self, priori):
        return self.measurement_function(priori)

    def compute_jacobian(self, function, *args):
        return nd.Jacobian(function)(*args)



if __name__ == "__main__":
    # dt = 0.02

    # position_funciton = lambda position, velocity : position + velocity*dt
    # velocity_funciton = lambda velocity, acceleration : velocity + acceleration*dt
    # acceleration_function = lambda acceleration : acceleration

    # def transition_function(state_vector):

    #     position = position_funciton(state_vector[0:2], state_vector[2:4])
    #     velocity = velocity_funciton(state_vector[2:4], state_vector[4:6])
    #     acceleration = acceleration_function(state_vector[4:6])

    #     return_state = np.hstack((position, velocity, acceleration))

    #     return np.reshape(return_state, (-1, 1))

    # def measurement_function(state_vector):
    #     return np.reshape(state_vector[0:2], (-1,1))

    # initial_state = np.asarray([1,2,3,4,5,6]).T

    # print(nd.Jacobian(transition_function)(initial_state))

    from test_script import test
    test(noise = 20, time_max = 3, show_acceleration_plots = False, filter_name = 'extended')
