import abc


class BaseBench(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def test(self, imgs):
        pass

    @abc.abstractmethod
    def model(self):
        pass

    @abc.abstractmethod
    def train(self, data, iteration=200, lr=1e-2, save_root='./checkpoints/'):
        pass
