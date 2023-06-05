import boto3
import botocore
from botocore import UNSIGNED
from botocore.client import Config
import os
import sys
from typing import List

sys.path.append("../../")

from util.log_utils import write_log
from util.parse_time import parseTime


ENDPOINT = "https://ams3.digitaloceanspaces.com"
REGION = "ams3"
BUCKET_NAME = "infoplaza-data-scientist-m-bruder"
OCEAN_CONFIG = Config(s3={"addressing_style": "virtual"}, signature_version=UNSIGNED)


class BucketService:
    """Used to query data from bucket"""

    def __init__(self, client=None, data_folder_path="../../../../data"):
        self.client = (
            client
            if client != None
            else boto3.client(
                "s3",
                endpoint_url=ENDPOINT,
                region_name=REGION,
                config=OCEAN_CONFIG,
            )
        )
        self.data_folder_path = data_folder_path

    def getFiles(self) -> List:
        response = self.client.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=1000)
        self.data = response["Contents"]
        while response["IsTruncated"]:
            continuation_token = response["NextContinuationToken"]
            response = self.client.list_objects_v2(
                Bucket=BUCKET_NAME, MaxKeys=1000, ContinuationToken=continuation_token
            )
            self.data.extend(response["Contents"])
        return self.data

    def downloadFile(self, key: str) -> None:
        self.client.download_file(BUCKET_NAME, key, f"{self.data_folder_path}/{key}")

    def downloadFilesInRange(self, time_span: tuple) -> bool:
        """
        Downloads files available in bucket.

        Parameters
        ----------
        `time_span`

        Raises
        ----------
        `Exception` when list of files is not available in self.data

        """
        if self.data == None or type(self.data) is not list:
            raise Exception("list of files does not exist. Query them with getFiles()")

        for obj in self.data:
            key = obj["Key"]
            # skip if already available:
            nat = key.replace("zip", "nat")
            if os.path.isfile(f"{self.data_folder_path}/{key}") or os.path.isfile(
                f"{self.data_folder_path}/{nat}"
            ):
                continue

            if time_span != None:
                time_object = parseTime(key)
                if (
                    time_object != None
                    and time_object > time_span[0]
                    and time_object < time_span[1]
                ):
                    self.downloadFile(key)
            else:
                self.downloadFile(key)

    def downloadAllFiles(self):
        return self.downloadFilesInRange(None)
