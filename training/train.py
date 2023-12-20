"""
Training pipeline of the neural network.

Author: Gijs G. Hendrickx
"""
import numpy as np
import torch
from sklearn import model_selection

from src import norm, model


class Training:
    """Training object, maintaining the same NN-model, loss function, optimizer, etc., throughout the training pipeline.

    This training class trains, tests, and saves an NN-model:
     -  train:  `.fit(x_train, y_train, epochs, **kwargs) -> loss_list`
     -  test:   `.test(x_test, y_test) -> loss`
     -  save:   `.save(file_name)`
    """

    def __init__(
            self, _model: torch.nn.Module, loss_function: torch.loss._Loss, optimizer: torch.optim.Optimizer, **kwargs
    ) -> None:
        """
        :param _model: NN-model
        :param loss_function: loss function
        :param optimizer: NN-optimizer
        :param kwargs: optional arguments
            device: device used for training, defaults to 'cpu'

        :type _model: torch.nn.Module
        :type loss_function: torch.loss._Loss
        :type optimizer: torch.optim.Optimizer
        :type kwargs: optional
            device: str
        """
        self.device: str = kwargs.get('device', 'cpu')

        self._model = _model.to(self.device)
        self.loss = loss_function
        self.opt = optimizer

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, **kwargs) -> list:
        """Fit (or train) the NN-model to a training data set for a number of epochs. This method follows supervised
        learning, i.e., the internal coefficients are updated to reduce the loss between the model-predicted output and
        the provided output (truth).

        :param x: input training data
        :param y: output training data
        :param epochs: number of training epochs
        :param kwargs: optional arguments
            random_sample_size: sample size of random selection, defaults to 10% of data

        :type x: numpy.ndarray
        :type y: numpy.ndarray
        :type epochs: int
        :type kwargs: optional
            random_sample_size: int

        :return: list of losses (per training epoch)
        :rtype: list

        :raise AssertionError: if `x` and `y` are not of equal length
        """
        assert len(x) == len(y), \
            f'Training input and output must be of equal length; input: {len(x)}, output: {len(y)}'
        n_train = len(x)

        # optional arguments
        random_sample_size: int = kwargs.get('random_sample_size', int(.1 * n_train))

        # training
        loss_list = []
        for ep in range(epochs):
            # reset optimizer
            self.opt.zero_grad()

            # randomly select samples from training data
            i_sel = np.random.choice(range(n_train), random_sample_size)
            x_tensor = torch.tensor(x[i_sel]).float().to(self.device)
            y_true = torch.tensor(y[i_sel]).float().to(self.device)

            # _model prediction
            y_model = self._model(x_tensor)

            # calculate loss
            loss = self.loss(y_model, y_true)

            # back-propagate
            loss.backward()
            self.opt.step()

            # append loss
            loss_list.append(float(loss.detach().cpu()))

        # return losses
        return loss_list

    def test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Test the NN-model to a testing data set. This method predicts the output with the latest state of the NN-
        model, and subsequently determines the loss w.r.t. the provided output data (i.e., the truth).

        :param x: input test data
        :param y: output test data

        :type x: numpy.ndarray
        :type y: numpy.ndarray

        :return: loss
        :rtype: float
        """
        x_tensor = torch.tensor(x).float().to(self.device)
        y_true = torch.tensor(y).float().to(self.device)

        y_model = self._model(x_tensor)

        loss = self.loss(y_model, y_true)
        loss = float(loss.detach().cpu())

        return loss


    def save(self, file_name: str) -> None:
        """Save the latest state of the NN-model.

        :param file_name: file name
        :type file_name: str
        """
        torch.save(self._model.state_dict(), file_name)


def train(_model: torch.nn.Module, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
    """Training sequence of the NN-model to prevent over-fitting (and under-fitting). This is achieved by increasing the
    number of training epochs and testing the NN-model performance at intermediate steps.

    :param _model: NN-model
    :param x: input data
    :param y: output data
    :param kwargs: optional arguments
        file_name: file name of NN-model, defaults to 'cann.pkl'
        loss_function: loss function, defaults to torch.nn.MSELoss
        max_epochs: maximum number of training epochs, defaults to 1000
        optimizer: optimizer function, defaults to torch.optim.Adam
        random_sample_size: sample size of random training selection, defaults to None
        step_epoch: epoch-increments per training step, defaults to 25
        test_size: relative size of test data, defaults to .2

    :type _model: torch.nn.Module
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :type kwargs: optional
        file_name: str
        loss_function: torch.loss._Loss
        max_epochs: int
        optimizer: torch.nn.Optimizer
        random_sample_size: int
        step_epoch: int
        test_size: float

    :return: best-fit model, error sequences of training and testing data sets
    :rtype: tuple

    :raise AssertionError: if `test_size` is not between 0 and 1
    """
    # optional arguments
    file_name: str = kwargs.get('file_name', 'cann.pkl')
    loss: torch.loss._Loss = kwargs.get('loss_function', torch.nn.MSELoss())
    max_epochs: int = kwargs.get('max_epochs', 1000)
    opt: torch.optim.Optimizer = kwargs.get('optimizer', torch.optim.Adam(_model.parameters()))
    rnd_size: int = kwargs.get('random_sample_size')
    step_epoch: int = kwargs.get('step_epoch', 25)
    test_size: float = kwargs.get('test_size', .2)

    # assert validity optional arguments
    assert 0 < test_size < 1, \
        f'Size of test data (`test_size`) must be between 0 and 1; {test_size} given'

    # normalised input/output data
    if not np.all((x >= 0) & (x <= 1)):
        x = norm.normalise(x)
    if not np.all((y >= 0) & (y <= 1)):
        y = norm.normalise(y)

    # split data: training v. testing
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size)

    # initiate training sequence
    obj_train = Training(_model, loss, opt)
    mse_train, mse_test = [], []
    epochs = step_epoch
    best_fit = epochs, 1e3, 1e3

    # training sequence
    while epochs < max_epochs:
        # train NN
        loss = obj_train.fit(x_train, y_train, step_epoch, random_sample_size=rnd_size)
        mse_train.append(loss[-1])
        mse_test.append(obj_train.test(x_test, y_test))

        # update best fit
        if mse_test[-1] < best_fit[-1]:
            best_fit = epochs, mse_train[-1], mse_test[-1]
            obj_train.save(file_name)

        # increment epochs
        epochs += step_epoch

    # return training metrics
    return obj_train._model, mse_train, mse_test


def prepare_data(file_name: str) -> tuple:
    pass


if __name__ == '__main__':
    inp, out = prepare_data('')
    nn_model, *mse = train(model._NNModel, inp, out)
