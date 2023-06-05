from datetime import datetime
import re
from functools import cmp_to_key


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


def order_based_on_file_timestamp(files: list) -> list:
    return sorted(files, key=cmp_to_key(compare_files))
