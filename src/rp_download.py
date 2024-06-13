import os
import uuid
import zipfile
from typing import List, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from botocore.config import Config
from boto3 import session
from boto3.s3.transfer import TransferConfig
import multiprocessing

HEADERS = {"User-Agent": "runpod-python/0.0.0 (https://runpod.io; support@runpod.io)"}

def calculate_chunk_size(file_size: int) -> int:
    '''
    Calculates the chunk size based on the file size.
    '''
    if file_size <= 1024*1024:  # 1 MB
        return 1024  # 1 KB
    if file_size <= 1024*1024*1024:  # 1 GB
        return 1024*1024  # 1 MB

    return 1024*1024*10  # 10 MB

def get_boto_client(bucket_creds: Optional[dict] = None) -> Tuple[boto3.client, TransferConfig]:
    '''
    Returns a boto3 client and transfer config for the bucket.
    '''
    bucket_session = session.Session()

    boto_config = Config(
        signature_version='s3v4',
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )

    transfer_config = TransferConfig(
        multipart_threshold=1024 * 25,
        max_concurrency=multiprocessing.cpu_count(),
        multipart_chunksize=1024 * 25,
        use_threads=True
    )

    if bucket_creds:
        endpoint_url = bucket_creds['endpointUrl']
        access_key_id = bucket_creds['accessId']
        secret_access_key = bucket_creds['accessSecret']
        region = bucket_creds['region']
    else:
        endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL', None)
        access_key_id = os.environ.get('BUCKET_ACCESS_KEY_ID', None)
        secret_access_key = os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
        region = os.getenv('BUCKET_REGION')

    if endpoint_url and access_key_id and secret_access_key:
        try:
            boto_client = bucket_session.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=boto_config,
                region_name=region
            )
        except NoCredentialsError:
            raise NoCredentialsError("Credentials not available.")
        except PartialCredentialsError:
            raise PartialCredentialsError("Incomplete credentials provided.")
    else:
        raise NoCredentialsError("No credentials provided in environment variables or bucket_creds.")

    return boto_client, transfer_config

def download_files_from_s3(
        job_id: str,
        bucket_name: str,
        object_names: Union[str, List[str]],
        bucket_creds: Optional[dict] = None) -> List[Optional[str]]:
    """
    Accepts a single S3 object name or a list of object names and downloads the files.
    Returns the list of downloaded file absolute paths.
    Saves the files in a directory called "downloaded_files" in the job directory.
    """
    download_directory = os.path.abspath(os.path.join('jobs', job_id, 'downloaded_files'))
    os.makedirs(download_directory, exist_ok=True)

    s3_client, transfer_config = get_boto_client(bucket_creds)

    @backoff.on_exception(backoff.expo, ClientError, max_tries=3)
    def download_file(s3_client, bucket_name: str, object_name: str, path_to_save: str):
        try:
            s3_client.download_file(bucket_name, object_name, path_to_save)
            print(f'File {bucket_name}/{object_name} downloaded to {path_to_save}')
        except Exception as e:
            print(f'Error downloading file: {e}')
            return None

    def download_file_to_path(object_name: str) -> Optional[str]:
        if object_name is None:
            return None

        file_extension = os.path.splitext(object_name)[1]
        file_name = f'{uuid.uuid4()}{file_extension}'
        output_file_path = os.path.join(download_directory, file_name)

        try:
            download_file(s3_client, bucket_name, object_name, output_file_path)
        except Exception as err:
            print(f"Failed to download {object_name}: {err}")
            return None

        return os.path.abspath(output_file_path)

    if isinstance(object_names, str):
        object_names = [object_names]

    if not object_names:
        return []

    with ThreadPoolExecutor() as executor:
        downloaded_files = list(executor.map(download_file_to_path, object_names))

    return downloaded_files

def file_s3(bucket_name: str, object_name: str) -> Optional[dict]:
    '''
    Downloads a single file from a given S3 bucket and object name, file is given a random name.
    If the file is a zip file, it is extracted into a directory with the same name.

    Returns an object that contains:
    - The absolute path to the downloaded file
    - File type
    - Original file name
    - Extracted path if it was a zip file
    '''
    os.makedirs('job_files', exist_ok=True)

    s3_client, _ = get_boto_client()

    file_extension = os.path.splitext(object_name)[1].replace('.', '')
    file_name = f'{uuid.uuid4()}'
    output_file_path = os.path.join('job_files', f'{file_name}.{file_extension}')

    try:
        s3_client.download_file(bucket_name, object_name, output_file_path)
    except Exception as e:
        print(f'Error downloading file: {e}')
        return None

    if file_extension == 'zip':
        unziped_directory = os.path.join('job_files', file_name)
        os.makedirs(unziped_directory, exist_ok=True)
        with zipfile.ZipFile(output_file_path, 'r') as zip_ref:
            zip_ref.extractall(unziped_directory)
        unziped_directory = os.path.abspath(unziped_directory)
    else:
        unziped_directory = None

    return {
        "file_path": os.path.abspath(output_file_path),
        "type": file_extension,
        "original_name": object_name,
        "extracted_path": unziped_directory
    }
