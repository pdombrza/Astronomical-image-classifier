from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def serve(self, input_image) -> dict[str, float]:
        ...


class AstronomicalClassifier(Model):
    def __init__(self, model):
        ...

    def serve(self, input_image) -> dict[str, float]:
        ...


class ModelStub(Model):
    def __init__(self):
        self.classes = [
            "Asteroid",
            "Black Hole",
            "Comet",
            "Galaxy",
            "Nebula",
            "Planet",
            "Star",
        ]

    def serve(self, input_image) -> dict[str, float]:
        mu = 3
        sigma = 1.5
        indices = np.arange(len(self.classes))
        probabilities = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (indices - mu)**2 / sigma**2)
        return {cl: prob for cl, prob in zip(self.classes, probabilities)}
