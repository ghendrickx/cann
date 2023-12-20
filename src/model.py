import numpy as np
import torch

from src import norm


class NNModel(torch.nn.Module):
    """Architecture of neural network building upon the `Module`-class of PyTorch."""
    _n_data = 250

    def __init__(self) -> None:
        """Initiation of the object auto-creates a pre-defined model-architecture."""
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(self._n_data, self._n_data),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self._n_data, self._n_data),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self._n_data, self._n_data),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self._n_data, self._n_data),
        )

    def forward(self, x):
        """Forward passing of neural network.

        :param x: input data
        :type x: torch.tensor

        :return: output data
        :rtype: torch.tensor
        """
        return self.features(x)


class CANN:
    """Cross-profile Artificial Neural Network."""

    def __init__(self, f_model: str, device: str = 'cpu') -> None:
        """
        :param f_model: file name of model (*.pkl)
        :param device: device, defaults to 'cpu'

        :type f_model: str
        :type device: str, optional
        """
        self._model = self._import_model(NNModel(), f_model)
        self._device = device

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Execute model.

        :param data: input data
        :type data: numpy.ndarray

        :return: model predictions
        :rtype: numpy.ndarray
        """
        return self.predict(data)

    @staticmethod
    def _import_model(model: torch.nn.Module, file: str) -> torch.nn.Module:
        """Import pre-trained neural network.

        :param model: model architecture
        :param file: file name

        :type model: torch.nn.Module
        :type file: str

        :return: pre-trained model
        :rtype: torch.nn.Module
        """
        model.load_state_dict(torch.load(file))
        return model

    @property
    def model(self) -> torch.nn.Module:
        """Neural network model.

        :return: model
        :rtype: torch.nn.Module
        """
        return self._model

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict with neural network model.

        :param data: input data
        :type data: numpy.ndarray

        :return: predicted output data
        :rtype: numpy.ndarray
        """
        # normalise input
        norm_data = norm.normalise(data)

        # execute NN
        x = torch.tensor(norm_data).float().to(self._device)
        y = self.model(x)

        # reverse normalise output
        out = norm.reverse(y.detach().cpu())

        # return output
        return out
