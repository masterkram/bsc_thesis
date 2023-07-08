from enum import Enum


class DatasetType(Enum):
    Sequence = 1
    SlidingWindow = 2
    ClassSequence = 3
    ClassSlidingWindow = 4
