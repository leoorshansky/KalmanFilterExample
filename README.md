# KalmanFilterExample
Example application of Kalman filter to predict an unknown 1-dimensional quantity.

Inspiration from [Cornell Computer Science](https://www.cs.cornell.edu/courses/cs4758/2012sp/materials/MI63slides.pdf)

## Usage

`python3 examples.py example1 -q 0.0001`

Runs Example 1 with the constant-level model

## Example Output

![Constant Level](/images/constant_model.png)

As we can see, the filter lags behind the measurements because it doesn't know that the level is increasing. It does a good job of reducing the noise, though.

`examples.py example1 -q 0.0001 -m increasing`

![Increasing Level](/images/increasing_model.png)

Here, the filter is given the information that it should expect an increasing water level, and therefore it does a good job smoothing the measurements and keeping up.

`examples.py example1 -q 0.01 -m increasing`

![High Process Noise](/images/high_process_noise.png)

Here, I have increased the expected process noise so much that the filter essentially trusts each new measurement at 100%. This shows the trade-off between trusting the model and trusting the measurement tool.

### I am currently working on implementing an extended kalman filter (EKF) for this example.