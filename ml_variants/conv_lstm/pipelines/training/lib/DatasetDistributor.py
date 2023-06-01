import math


class DatasetDistributor:
    def __init__(self, file_quantity: int, splits: list):
        self.file_quantity = file_quantity
        self.splits = splits
        self.index = 0
        self.distribution_position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.splits) <= self.index:
            raise StopIteration(
                f"not enough split parameters expected {self.index} but was {len(self.splits)}"
            )

        increase = math.floor(self.file_quantity * self.splits[self.index])
        upper_limit = self.distribution_position + increase

        if upper_limit > self.file_quantity:
            raise StopIteration(
                f"not enough files to satisfy split, wanted to allocate until {upper_limit}, but file amount is {self.file_quantity}"
            )

        lower_limit = self.distribution_position
        self.distribution_position = upper_limit
        self.index += 1

        return (lower_limit, upper_limit)
