from datetime import datetime
import re


def parseTime(key: str) -> datetime:
    date = re.search("([0-9]){12}", key)
    if date == None:
        return None
    date = date.group()
    return datetime.strptime(date, "%Y%m%d%H%M")
