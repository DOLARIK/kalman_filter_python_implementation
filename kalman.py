import numpy as np

class Kalman:

    def __init__(self, initial_state, initial_error_covariance_matrix,
                        transition_matrix, process_noise,
                        input_matrix, input_vector,
                        measurement_matrix, measurement_noise):

        self.initial_state = initial_state
        self.initial_error_covariance_matrix = initial_error_covariance_matrix

        self.transition_matrix = transition_matrix
        self.process_noise = process_noise

        self.input_matrix = input_matrix
        self.input_vector = input_vector

        self.measurement_matrix = measurement_matrix
        self.measurement_noise = measurement_noise

        self.prioris = []
        self.error_covariance = initial_error_covariance_matrix

        self.posterioris = []
        self.posterioris.append(initial_state)

    def __call__(self, sensor_readings):

        # PRIORI CALCULATION, PRIORI_ERROR_COVARIANCE
        priori, predicted_error_covariance = self.priori_calculation(
                                                        state_vector = self.posterioris[-1],
                                                        error_covariance = self.error_covariance)

        self.prioris.append(priori)

        # OBSERVE READINGS, EXPECTED_READINGS, EXPECTED_ERROR_COVARIANCE
        expected_readings, expected_error_covariance = self.expectations(priori = priori,
                                                                    predicted_error_covariance = predicted_error_covariance)

        # KALMAN_GAIN
        kalman_gain_matrix = self.kalman_gain(predicted_error_covariance, expected_error_covariance)

        # POSTERIORI CALCULATION
        posteriori = self.posteriori_calculation(
                        sensor_readings = sensor_readings,
                        kalman_gain_matrix = kalman_gain_matrix,
                        expected_readings = expected_readings,
                        predicted_error_covariance = predicted_error_covariance)

        self.posterioris.append(posteriori)

        return priori, posteriori


    def priori_calculation(self, state_vector, error_covariance):

        priori = np.matmul(self.transition_matrix, state_vector.reshape(state_vector.size, 1)) + self.input_matrix.reshape(state_vector.size, self.input_vector.shape[0])*self.input_vector
        predicted_error_covariance = np.matmul(np.matmul(self.transition_matrix,error_covariance), self.transition_matrix.T) + self.process_noise

        return priori, predicted_error_covariance

    def expectations(self, priori, predicted_error_covariance):

        expected_readings = np.matmul(self.measurement_matrix, priori)
        expected_error_covariance = np.matmul(np.matmul(self.measurement_matrix,predicted_error_covariance),self.measurement_matrix.T)

        return expected_readings, expected_error_covariance

    def kalman_gain(self, predicted_error_covariance, expected_error_covariance):

        kalman_gain_matrix = np.matmul(np.matmul(predicted_error_covariance, self.measurement_matrix.T), np.linalg.inv(expected_error_covariance + self.measurement_noise))

        return kalman_gain_matrix

    def posteriori_calculation(self, sensor_readings, kalman_gain_matrix,
                                    expected_readings, predicted_error_covariance):

        posteriori = self.prioris[-1] + np.matmul(kalman_gain_matrix, (sensor_readings.reshape(sensor_readings.size, 1) - expected_readings))
        updated_error_covariance = predicted_error_covariance - np.matmul(kalman_gain_matrix, np.matmul(self.measurement_matrix, predicted_error_covariance))

        self.error_covariance = updated_error_covariance

        return posteriori
