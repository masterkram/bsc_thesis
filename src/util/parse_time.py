from datetime import datetime
import re
from functools import cmp_to_key
from typing import List


def parseTime(key: str) -> datetime:
    date = re.search("([0-9]){12}", key)
    if date == None:
        return None
    date = date.group()
    return datetime.strptime(date, "%Y%m%d%H%M")


def compare_files(file1: str, file2: str) -> int:
    date1, date2 = parseTime(file1), parseTime(file2)
    if date1 > date2:
        return 1
    elif date1 < date2:
        return -1

    return 0


def order_based_on_file_timestamp(files: List) -> List:
    return sorted(files, key=cmp_to_key(compare_files))


def find_matching_time(array: List, date: datetime) -> int:
    """
    TODO: Add Binary Search
    """
    for i in range(len(array)):
        current_time = parseTime(array[i])
        # print(current_time, "vs", date)

        if current_time >= date:
            return i


def get_next_sequence(sequence_length: int, selection: str, array: List) -> List:
    start_time = parseTime(selection)
    print(start_time)
    start_index = find_matching_time(array, start_time)
    return (start_index, start_index + sequence_length)
