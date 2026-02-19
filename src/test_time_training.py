"""Base interface for TTT methods."""

from dataclasses import dataclass


@dataclass
class TTTConfig:
    lr: float = 1e-3
    steps_per_batch: int = 1
    adaptation_task: str = "rotation"


class TTTAdapter:
    """Minimal TTT adapter skeleton.

    Replace TODO blocks with your actual model update logic.
    """

    def __init__(self, config: TTTConfig):
        self.config = config

    def adapt_on_batch(self, batch):
        """Adapt model parameters using unlabeled target batch."""
        # TODO: implement self-supervised adaptation objective.
        return batch

    def predict(self, batch):
        """Run inference after adaptation."""
        # TODO: replace with actual model forward pass.
        return batch
