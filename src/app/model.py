from abc import ABC, abstractmethod
import numpy as np
import torch
import torchvision.transforms as transforms


class Model(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def serve(self, input_image) -> dict[str, float]:
        ...


class AstronomicalClassifier(Model):
    def __init__(self, model, weights_path):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        input_image_tensor = transform(input_image).unsqueeze(0).to(self.device)
        output = self.model(input_image_tensor)
        output = torch.squeeze(output)
        return {cl: prob for cl, prob in zip(self.classes, output)}


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
