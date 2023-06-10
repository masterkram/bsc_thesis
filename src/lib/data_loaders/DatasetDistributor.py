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


class DatasetDistributorCombined:
    def __init__(
        self, file_quantity_satellite: int, file_quantity_radar: int, splits: list
    ):
        self.file_quantity_satellite = file_quantity_satellite
        self.file_quantity_radar = file_quantity_radar
        self.splits = splits
        self.index = 0
        self.distribution_position_satellite = 0
        self.distribution_position_radar = 0
        self.to_radar_resolution = lambda sat: math.floor(
            sat * (file_quantity_radar / file_quantity_satellite)
        )

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.splits) <= self.index:
            raise StopIteration(
                f"not enough split parameters expected {self.index} but was {len(self.splits)}"
            )

        increase_satellite = math.floor(
            self.file_quantity_satellite * self.splits[self.index]
        )
        increase_radar = self.to_radar_resolution(increase_satellite)

        upper_limit_satellite = (
            self.distribution_position_satellite + increase_satellite
        )
        upper_limit_radar = self.distribution_position_radar + increase_radar

        if upper_limit_satellite > self.file_quantity_satellite:
            raise StopIteration(
                f"not enough files to satisfy split, wanted to allocate until {upper_limit_satellite}, but file amount is {self.file_quantity_satellite}"
            )

        if upper_limit_radar > self.file_quantity_radar:
            raise StopIteration(
                f"not enough files to satisfy split, wanted to allocate until {upper_limit_radar}, but file amount is {self.file_quantity_radar}"
            )

        lower_limit_satellite = self.distribution_position_satellite
        lower_limit_radar = self.distribution_position_radar

        self.distribution_position_satellite = upper_limit_satellite
        self.distribution_position_radar = upper_limit_radar
        self.index += 1

        return (lower_limit_satellite, upper_limit_satellite), (
            lower_limit_radar,
            upper_limit_radar,
        )
