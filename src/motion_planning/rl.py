from typing import Any, Iterator, Protocol

from torch.utils.data import IterableDataset


class SampleGenerator(Protocol):
    def __call__(self, *args, **kwargs) -> Iterator[Any]: ...


class ExperienceSourceDataset(IterableDataset):
    """Basic experience source dataset.
    Takes a generate_batch function that returns an iterator. The logic for the experience source and how the batch is
    generated is defined the Lightning model itself
    """

    def __init__(self, sample_generator: SampleGenerator, length: int, *args, **kwargs):
        self.sample_generator = sample_generator
        self.length = length
        self.args = args
        self.kwargs = kwargs

    def __iter__(self) -> Iterator:
        return self.sample_generator(*self.args, **self.kwargs)

    def __len__(self) -> int:
        return self.length
