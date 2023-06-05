import boto3
import botocore
from botocore import UNSIGNED
from botocore.client import Config
import os
import sys
from typing import List
import typed_settings as ts
from tqdm import tqdm

sys.path.append("../../")

from Settings import DownloadSettings

from util.log_utils import write_log
from util.parse_time import parseTime

OCEAN_CONFIG = Config(s3={"addressing_style": "virtual"}, signature_version=UNSIGNED)

settings = ts.load(DownloadSettings, "downloads", config_files=["config.toml"])


class BucketService:
    """Used to query data from bucket"""

    def __init__(self, client=None, data_folder_path="../../../../data"):
        self.client = (
            client
            if client != None
            else boto3.client(
                "s3",
                endpoint_url=settings.endpoint,
                region_name=settings.region,
                config=OCEAN_CONFIG,
            )
        )
        self.data_folder_path = data_folder_path

    def getFiles(self) -> List:
        response = self.client.list_objects(Bucket=settings.bucket_name)
        self.data = response["Contents"]
        while response["IsTruncated"]:
            continuation_token = response["NextContinuationToken"]
            response = self.client.list_objects_v2(
                Bucket=settings.bucket_name,
                MaxKeys=1000,
                ContinuationToken=continuation_token,
            )
            self.data.extend(response["Contents"])
        return self.data

    def downloadFile(self, key: str) -> None:
        splitKey = key.split("/")

        if len(splitKey) != 5 or len(splitKey[-1]) == 0:
            return

        if "radar" in splitKey[0]:
            splitKey[0] = "radar"
        elif "satellite" in splitKey[0]:
            splitKey[0] = "satellite"

        return self.client.download_file(
            settings.bucket_name,
            key,
            os.path.join(self.data_folder_path, splitKey[0], splitKey[1]),
        )

    def downloadFilesInRange(self, time_span: tuple, loadBar=False) -> bool:
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

        for obj in tqdm(self.data):
            key = obj["Key"]
            # skip if already available:
            nat = key.replace("zip", "nat")
            if os.path.isfile(f"{self.data_folder_path}/{key}") or os.path.isfile(
                f"{self.data_folder_path}/{nat}"
            ):
                continue

            if time_span is not None:
                time_object = parseTime(key)

                if (
                    time_object != None
                    and time_object > time_span[0]
                    and time_object < time_span[1]
                ):
                    self.downloadFile(key)
            else:
                self.downloadFile(key)

    def downloadAllFiles(self, loadBar=False):
        return self.downloadFilesInRange(None, loadBar)
