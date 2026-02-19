"""Base interface for SSL methods."""

from dataclasses import dataclass


@dataclass
class SSLConfig:
    method: str = "simclr"
    epochs: int = 100
    batch_size: int = 128


class SelfSupervisedTrainer:
    """Minimal SSL trainer skeleton."""

    def __init__(self, config: SSLConfig):
        self.config = config

    def pretrain(self, train_loader):
        """Run self-supervised pretraining."""
        # TODO: implement SSL pretraining loop.
        return train_loader

    def linear_eval(self, val_loader):
        """Run linear evaluation on frozen features."""
        # TODO: implement evaluation pipeline.
        return val_loader
