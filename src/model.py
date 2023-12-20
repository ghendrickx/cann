import torch

from src import norm


class _NNModel(torch.nn.Module):
    """Architecture of neural network building upon the `Module`-class of PyTorch."""

    def __init__(self) -> None:
        """Initiation of the object auto-creates a pre-defined model-architecture."""
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1000, 1000),
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

    def __init__(self, f_model: str, device: str = 'cpu') -> None:
        self._model = self._import_model(_NNModel(), f_model)
        self._device = device

    def __call__(self, data):
        return self.predict(data)

    @staticmethod
    def _import_model(model: torch.nn.Module, file: str) -> torch.nn.Module:
        model.load_state_dict(torch.load(file))
        return model

    @property
    def model(self):
        return self._model

    def predict(self, data):
        # normalise input
        norm_data = norm.normalise(data)

        # execute NN
        x = torch.tensor(norm_data).float().to(self._device)
        y = self.model(x)

        # reverse normalise output
        out = norm.reverse(y.detach().cpu())

        # return output
        return out
