import click
import numpy as np

@click.group()
def cli ():
    pass

class KalmanFilter: # Class which represents the Kalman Filter

    def __init__ (self, dim, x_initial, p_initial, state_transition, process_error, measure_error, measure_matrix, extended): # Assign initial values
        self.dim = dim
        self.x = np.reshape(x_initial, (dim, 1))
        self.p = p_initial
        self.state_transition = state_transition
        self.process_error = process_error
        self.measure_error = measure_error
        self.measure_matrix = np.reshape(measure_matrix, (-1, dim))
        self.extended = extended

    def predict_and_update (self, measurement, hjacobian = None, hx = None): # Perform an iteration of the predict and update steps. Returns new state estimate

        if self.dim == 1: # Scalar multiplication instead of matrix multiplication
            # PREDICT
            x_prime = self.state_transition * self.x
            p_prime = self.state_transition * self.p * self.state_transition.T + self.process_error

            # UPDATE
            k = p_prime * self.measure_matrix.T * np.linalg.inv(np.reshape(self.measure_matrix * p_prime * self.measure_matrix.T + self.measure_error, (1, 1)))
            self.x = x_prime + k * (measurement - self.measure_matrix * x_prime)
            self.p = (np.identity(self.dim) - k * self.measure_matrix) * p_prime

            return self.x

        # @ denotes matrix multiplication in numpy

        if self.extended:
            if hjacobian is None and hx is None:
                raise Exception("Need HJacobian and h(x) functions for EKF")
            self.measure_matrix = hjacobian(self.x)

        # PREDICT
        x_prime = self.state_transition @ self.x
        p_prime = self.state_transition @ self.p @ self.state_transition.T + self.process_error

        # UPDATE
        k = np.reshape(p_prime @ self.measure_matrix.T @ np.linalg.inv(np.reshape(self.measure_error + self.measure_matrix @ p_prime @ self.measure_matrix.T, (1, 1))), (self.dim, 1))
        if not self.extended:
            self.x = x_prime + k @ np.reshape(measurement - self.measure_matrix @ x_prime, (1, self.dim))
        else:
            self.x = x_prime + k @ np.reshape(measurement - hx(x_prime), (1, self.dim))
        self.p = (np.eye(self.dim) - k @ self.measure_matrix) @ p_prime

        return self.x

def array_from_string(s): # Converts argument string into numpy array
    if s.strip()[1:-1].find("[") == -1:
        return np.fromstring(s.strip()[1:-1], sep=' ')
    else:
        s = s.strip()[1:-1]
        count = 0
        last = 0
        subarrays = []
        for i in range(len(s)):
            if s[i] == "[":
                if count == 0:
                    last = i
                count += 1
            elif s[i] == "]":
                count -= 1
                if count == 0:
                    subarrays.append(s[last:i+1])
        return np.stack([array_from_string(x) for x in subarrays])

@cli.command()
@click.option('-df', '--data-file', required=True, help="Path to Data File (stored as text file with observation vector on each line)")
@click.option('-d', '--dimension', required=True, help="Dimensionality of Environment State")
@click.option('-x0', '--initial-guess', required=True, help="Assumed State at Time Zero (1 x d) in the form [0 0] etc.")
@click.option('-p0', '--initial-error', required=True, help="Assumed Initial Error Matrix (d x d) in the form [[1 0] [0 1]] etc.")
@click.option('-f', '--transition-matrix', required=True, help="State Transition Matrix (d x d) in the form [[1 0] [0 1]] etc.")
@click.option('-q', '--process-error-matrix', required=True, help="Process Error Matrix (d x d) in the form [[1 0] [0 1]] etc.")
@click.option('-r', '--measurement-error-matrix', required=True, help="Measurement Error Matrix (d x d) in the form [[1 0] [0 1]] etc.")
@click.option('-h', '--measurement-matrix', default=np.array(1), show_default=True, help="Measurement Matrix (? x d) in the form [[1 0] [0 1]] etc.")
@click.option('-e', '--extended', default=False, show_default=True, help="Is Extended Kalman Filter")
def kalman (data_file, dimension, initial_guess, initial_error, transition_matrix, process_error_matrix, measurement_error_matrix, measurement_matrix, extended, hjacobian = None, hx = None, _cli=True):
    # Command Line Interface
    if _cli:
        dimension = int(dimension)
        initial_guess = array_from_string(initial_guess)
        initial_error = array_from_string(initial_error)
        transition_matrix = array_from_string(transition_matrix)
        process_error_matrix = array_from_string(process_error_matrix)
        measurement_error_matrix = array_from_string(measurement_error_matrix)
        measurement_matrix = array_from_string(measurement_matrix)
        kf =  KalmanFilter(dimension, initial_guess, initial_error, transition_matrix, process_error_matrix, measurement_error_matrix, measurement_matrix, False)
        with open(data_file, 'r') as openfile:
            for line in openfile:
                print(kf.predict_and_update(array_from_string(line)))
        return
    kf =  KalmanFilter(dimension, initial_guess, initial_error, transition_matrix, process_error_matrix, measurement_error_matrix, measurement_matrix, extended)
    estimates = np.empty((0, dimension))
    for measurement in data_file:
        estimate = kf.predict_and_update(measurement, hjacobian, hx)[:, 0]
        estimates = np.append(estimates, [estimate], 0)
    return estimates



if __name__ == "__main__": # Run command line parser
    cli()