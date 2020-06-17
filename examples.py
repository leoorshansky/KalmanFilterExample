from kalman import kalman
import numpy as np
import click
import matplotlib.pyplot as plt

@click.group()
def cli():
    pass

# Simple Example
@cli.command()
@click.option("-q", "--process-error", required=True, default="0.0001", show_default=True, help="Try Tuning This and See What Happens")
@click.option("-m", "--model", type=click.Choice(["constant", "increasing"]), required=True, default="constant", show_default=True, help="Constant vs. Increasing Model")
@click.pass_context
def example1 (ctx, process_error, model):
    if model == "constant":
        process_error = np.fromstring(process_error, sep=" ")
        d = 1
        df = np.reshape(np.arange(3, 5.9, 0.1) + (np.random.random_sample(30) - 0.5) * 0.4 + 0.3 * np.sin(2 * np.pi * 0.13 * np.arange(0, 30)), (30, 1))
        f = np.array(1)
        x = np.array(0)
        p = np.array(1000)
        r = np.array(0.01)
        h = np.array(1)
    elif model == "increasing":
        process_error = np.fromstring(process_error, sep=" ") * np.array([[1/3, 1/2], [1/2, 1]])
        d = 2
        df = np.stack([np.arange(3, 5.9, 0.1) + (np.random.random_sample(30) - 0.5) * 0.4 + 0.3 * np.sin(2 * np.pi * 0.13 * np.arange(0, 30)), np.zeros(30)], 1)
        f = np.array([[1, 1], [0, 1]])
        x = np.array([0, 0])
        p = np.array([[1000, 0], [0, 1000]])
        r = np.array(0.1)
        h = np.array([1, 0])
    estimate = ctx.invoke(kalman, 
        data_file=df, 
        dimension=d, 
        initial_guess=x, # Absolutely arbitrary
        initial_error=p, # High error because we have no clue what the initial value is
        transition_matrix=f, # State is either assumed constant or assumed to be gradually increasing
        process_error_matrix=process_error, # Relatively simple process, no need to think that state is changing much beyond what our model predicts
        measurement_error_matrix=r, # Relatively low measurement error
        measurement_matrix=h, # 1 to 1 map between measurement and state
        _cli=False)
    _, ax = plt.subplots()
    ax.plot(np.arange(0, 30), estimate[:,0], label="estimated")
    ax.plot(np.arange(0, 30), df[:,0], label="measured")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    cli()