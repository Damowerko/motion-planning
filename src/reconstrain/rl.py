import random
from torch.utils.data import IterableDataset
from typing import Any, Iterable, Iterator, List, Generic, TypeVar, Tuple
from typing import Protocol


class SampleGenerator(Protocol):
    def __call__(self, *args, **kwargs) -> Iterator[Any]:
        ...


class ExperienceSourceDataset(IterableDataset):
    """Basic experience source dataset.
    Takes a generate_batch function that returns an iterator. The logic for the experience source and how the batch is
    generated is defined the Lightning model itself
    """

    def __init__(self, sample_generator: SampleGenerator, *args, **kwargs):
        self.sample_generator = sample_generator
        self.args = args
        self.kwargs = kwargs

    def __iter__(self) -> Iterator:
        return self.sample_generator(*self.args, **self.kwargs)


T = TypeVar("T")


class ReplayBuffer(Generic[T]):
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, data: T):
        self.buffer[self.index] = data
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def extend(self, iterable: Iterable[T]):
        for data in iterable:
            self.append(data)

    def sample(self, num_samples) -> List[T]:
        indices = random.choices(range(self.size), k=num_samples)
        return [self.buffer[index] for index in indices]

    def collect(self, shuffle=False) -> List[T]:
        if shuffle:
            return self.sample(self.size)
        return self.buffer[: self.size]

    def __len__(self) -> int:
        return self.size


class WeightedReplayBuffer(ReplayBuffer[T]):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.weights = [0.0] * max_size

    def append(self, data: T, weight: float):
        self.weights[self.index] = weight
        super().append(data)

    def extend(self, iterable: Iterable[Tuple[T, float]]):
        for data, weight in iterable:
            self.append(data, weight)

    def sample(self, num_samples) -> List[T]:
        indices = random.choices(
            range(self.size), weights=self.weights[: self.size], k=num_samples
        )
        return [self.buffer[index] for index in indices]
