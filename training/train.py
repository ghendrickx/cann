import typing

import numpy as np
import torch
from sklearn import model_selection

from src import norm


class Training:

    def __init__(self, model, loss_function: torch.loss._Loss, optimizer: torch.optim.Optimizer, **kwargs) -> None:
        self.device = kwargs.get('device', 'cpu')

        self.model = model.to(self.device)
        self.loss = loss_function
        self.opt = optimizer

    def fit(self, x, y, epochs: int, **kwargs) -> list:
        assert len(x) == len(y), \
            f'Training input and output must be of equal length; input: {len(x)}, output: {len(y)}'
        n_train = len(x)

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

            # model prediction
            y_model = self.model(x_tensor)

            # calculate loss
            loss = self.loss(y_model, y_true)

            # back-propagate
            loss.backward()
            self.opt.step()

            # append loss
            loss_list.append(float(loss.detach().cpu()))

        # return losses
        return loss_list

    def test(self, x, y) -> float:
        x_tensor = torch.tensor(x).float().to(self.device)
        y_true = torch.tensor(y).float().to(self.device)

        y_model = self.model(x_tensor)

        loss = self.loss(y_model, y_true)
        loss = float(loss.detach().cpu())

        return loss


    def save(self, file_name: str) -> None:
        torch.save(self.model.state_dict(), file_name)


def train(model, x, y, **kwargs):
    # optional arguments
    file_name: str = kwargs.get('file_name', 'cann.pkl')
    loss: torch.loss._Loss = kwargs.get('loss_function', torch.nn.MSELoss())
    opt: torch.optim.Optimizer = kwargs.get('optimizer', torch.optim.Adam(model.parameters()))
    seq_epochs: typing.Sequence[int] = kwargs.get('seq_epochs', np.arange(100, 1001, step=25, dtype=int))
    rnd_size: int = kwargs.get('random_sample_size')
    test_size: float = kwargs.get('test_size', .2)

    # normalised input/output data
    if not np.all((x >= 0) & (x <= 1)):
        x = norm.normalise(x)
    if not np.all((y >= 0) & (y <= 1)):
        y = norm.normalise(y)

    # split data: training v. testing
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size)

    # initiate training sequence
    obj_train = Training(model, loss, opt)
    mse_train, mse_test = [], []
    best_fit = seq_epochs[0], 1e3, 1e3

    # training sequence
    for i, ep in enumerate(seq_epochs):
        # train NN
        loss = obj_train.fit(x_train, y_train, ep, random_sample_size=rnd_size)
        mse_train.append(loss[-1])
        mse_test.append(obj_train.test(x_test, y_test))
        
        # update best fit
        if mse_test[-1] < best_fit[-1]:
            best_fit = ep, mse_train[-1], mse_test[-1]
            obj_train.save(file_name)

    # return training metrics
    return mse_train, mse_test
