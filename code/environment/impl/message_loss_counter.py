from collections import defaultdict


class MessageLossCounter:
    def __init__(self):
        self.reset()

    @property
    def counts(self) -> defaultdict:
        return self._counts.copy()

    @property
    def total_counts(self) -> int:
        return self._total_counts

    def get_count(self, node_id: str):
        return self._counts[node_id]

    def register_loss(self, node_id: str, num_loss: int):
        self._counts[node_id] = self._counts[node_id] + num_loss
        self._total_counts += num_loss

    def reset(self):
        self._counts = defaultdict(lambda: 0)
        self._total_counts = 0
