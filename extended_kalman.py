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
            transition_matrix = self.compute_jacobian(transition_function, initial_state),
            process_noise = process_noise,
            input_matrix = input_matrix,
            input_vector = input_vector,
            measurement_matrix = self.compute_jacobian(measurement_function, initial_state),
            measurement_noise = measurement_noise)

    def __call__(self, sensor_readings):

        # PRIORI, POSTERIORI CALCULATION
        priori, posteriori = super().__call__(sensor_readings)

        # JACOBIAN TRANSITION AND MEASUREMENT MATRIX COMPUTATION AND UPDATION
        self.transition_matrix = self.compute_jacobian(self.transition_function, self.posterioris[-1])
        self.measurement_matrix = self.compute_jacobian(self.measurement_function, self.posterioris[-1])

        return priori, posteriori


    def compute_jacobian(self, function, state):
        return nd.Jacobian(fun)(state)


if __name__ == "__main__":
    pass
