import os
import uuid
import zipfile
import logging
from typing import List, Union, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import backoff
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from botocore.config import Config
from boto3 import session
from boto3.s3.transfer import TransferConfig
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def get_boto_client(bucket_creds: Optional[dict] = None, job_logs: Optional[List[str]] = None) -> Tuple[Optional[boto3.client], Optional[TransferConfig]]:
    '''
    Returns a boto3 client and transfer config for the bucket.
    '''
    if job_logs is None:
        job_logs = []

    bucket_session = session.Session()
    logger.info("Boto3 session created.")
    job_logs.append("INFO: Boto3 session created.")

    boto_config = Config(
        signature_version='s3v4',
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
    logger.info("Boto3 config initialized.")
    job_logs.append("INFO: Boto3 config initialized.")

    transfer_config = TransferConfig(
        multipart_threshold=1024 * 25,
        max_concurrency=multiprocessing.cpu_count(),
        multipart_chunksize=1024 * 25,
        use_threads=True
    )
    logger.info("Boto3 transfer config initialized.")
    job_logs.append("INFO: Boto3 transfer config initialized.")

    if bucket_creds:
        endpoint_url = bucket_creds.get('endpointUrl')
        access_key_id = bucket_creds.get('accessId')
        secret_access_key = bucket_creds.get('accessSecret')
        region = bucket_creds.get('region')
        logger.info("Using provided bucket credentials.")
        job_logs.append("INFO: Using provided bucket credentials.")
    else:
        endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL')
        access_key_id = os.environ.get('BUCKET_ACCESS_KEY_ID')
        secret_access_key = os.environ.get('BUCKET_SECRET_ACCESS_KEY')
        region = os.getenv('BUCKET_REGION')
        logger.info("Using environment variables for bucket credentials.")
        job_logs.append("INFO: Using environment variables for bucket credentials.")

    if endpoint_url and access_key_id and secret_access_key:
        try:
            logger.info(f"Attempting to create S3 client with endpoint: {endpoint_url}, region: {region}")
            job_logs.append(f"INFO: Attempting to create S3 client with endpoint: {endpoint_url}, region: {region}")
            boto_client = bucket_session.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=boto_config,
                region_name=region
            )
            logger.info("S3 client created successfully.")
            job_logs.append("INFO: S3 client created successfully.")
            return boto_client, transfer_config
        except NoCredentialsError as e:
            logger.error(f"NoCredentialsError: Credentials not available. Details: {e}")
            job_logs.append(f"ERROR: NoCredentialsError: Credentials not available. Details: {e}")
            return None, None
        except PartialCredentialsError as e:
            logger.error(f"PartialCredentialsError: Incomplete credentials provided. Details: {e}")
            job_logs.append(f"ERROR: PartialCredentialsError: Incomplete credentials provided. Details: {e}")
            return None, None
        except ClientError as e:
            logger.error(f"ClientError creating S3 client: {e}")
            job_logs.append(f"ERROR: ClientError creating S3 client: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error creating S3 client: {e}")
            job_logs.append(f"ERROR: Unexpected error creating S3 client: {e}")
            return None, None
    else:
        creds_detail = (
            f"Endpoint URL: {'Provided' if endpoint_url else 'Missing'}, "
            f"Access Key ID: {'Provided' if access_key_id else 'Missing'}, "
            f"Secret Access Key: {'Provided' if secret_access_key else 'Missing'}"
        )
        logger.error(f"No valid credentials provided in environment variables or bucket_creds. Details: {creds_detail}")
        job_logs.append(f"ERROR: No valid credentials provided. Details: {creds_detail}")
        return None, None


def download_files_from_s3(
        job_id: str,
        bucket_name: str,
        object_names: Union[str, List[str]],
        bucket_creds: Optional[dict] = None) -> Tuple[List[Optional[str]], List[str]]:
    """
    Accepts a single S3 object name or a list of object names and downloads the files.
    Returns the list of downloaded file absolute paths and a list of log messages.
    Saves the files in a directory called "downloaded_files" in the job directory.
    """
    job_logs: List[str] = []
    download_directory = os.path.abspath(os.path.join('jobs', job_id, 'downloaded_files'))
    try:
        os.makedirs(download_directory, exist_ok=True)
        logger.info(f"Job {job_id}: Created download directory: {download_directory}")
        job_logs.append(f"INFO: Job {job_id}: Created download directory: {download_directory}")
    except OSError as e:
        logger.error(f"Job {job_id}: Error creating download directory {download_directory}: {e}")
        job_logs.append(f"ERROR: Job {job_id}: Error creating download directory {download_directory}: {e}")
        return [], job_logs

    s3_client, transfer_config = get_boto_client(bucket_creds, job_logs)
    if not s3_client:
        logger.error(f"Job {job_id}: Failed to get S3 client. Aborting download.")
        job_logs.append(f"ERROR: Job {job_id}: Failed to get S3 client. Aborting download.")
        return [], job_logs

    @backoff.on_exception(backoff.expo, ClientError, max_tries=3)
    def download_file_with_retry(s3_client_instance, current_bucket_name: str, current_object_name: str, path_to_save: str, current_job_logs: List[str]):
        try:
            log_msg = f"Job {job_id}: Attempting to download {current_bucket_name}/{current_object_name} to {path_to_save}"
            logger.info(log_msg)
            current_job_logs.append(f"INFO: {log_msg}")
            s3_client_instance.download_file(current_bucket_name, current_object_name, path_to_save, Config=transfer_config)
            success_msg = f"Job {job_id}: File {current_bucket_name}/{current_object_name} downloaded successfully to {path_to_save}"
            logger.info(success_msg)
            current_job_logs.append(f"INFO: {success_msg}")
            return True
        except ClientError as e:
            error_msg = f"Job {job_id}: ClientError downloading {current_object_name} (attempt {e.kwargs.get('attempt_number', 'N/A') if hasattr(e, 'kwargs') else 'N/A'}): {e}"
            logger.warning(error_msg) # Warning for backoff to retry
            current_job_logs.append(f"WARNING: {error_msg}")
            raise  # Re-raise for backoff
        except Exception as e:
            error_msg = f"Job {job_id}: Unexpected error downloading file {current_object_name}: {e}"
            logger.error(error_msg)
            current_job_logs.append(f"ERROR: {error_msg}")
            return False # Indicate failure for non-ClientError exceptions

    def download_file_to_path(object_name: str) -> Optional[str]:
        local_job_logs: List[str] = [] # Logs specific to this thread/file download
        if object_name is None:
            log_msg = f"Job {job_id}: Object name is None, skipping download."
            logger.warning(log_msg)
            local_job_logs.append(f"WARNING: {log_msg}")
            job_logs.extend(local_job_logs)
            return None

        file_extension = os.path.splitext(object_name)[1]
        file_name_uuid = f'{uuid.uuid4()}{file_extension}'
        output_file_path = os.path.join(download_directory, file_name_uuid)

        log_msg = f"Job {job_id}: Preparing to download {object_name} to {output_file_path}"
        logger.info(log_msg)
        local_job_logs.append(f"INFO: {log_msg}")

        try:
            if download_file_with_retry(s3_client, bucket_name, object_name, output_file_path, local_job_logs):
                job_logs.extend(local_job_logs)
                return os.path.abspath(output_file_path)
            else:
                # download_file_with_retry already logged the error
                job_logs.extend(local_job_logs)
                return None
        except Exception as err: # Catch exceptions from backoff if all retries fail
            error_msg = f"Job {job_id}: Failed to download {object_name} after multiple retries: {err}"
            logger.error(error_msg)
            local_job_logs.append(f"ERROR: {error_msg}")
            job_logs.extend(local_job_logs)
            return None

    if isinstance(object_names, str):
        object_names = [object_names]

    if not object_names:
        logger.info(f"Job {job_id}: No object names provided for download.")
        job_logs.append(f"INFO: Job {job_id}: No object names provided for download.")
        return [], job_logs

    downloaded_files: List[Optional[str]] = []
    # Using ThreadPoolExecutor might complicate log aggregation if not careful.
    # For simplicity in log passing, let's process sequentially or ensure logs are correctly passed back.
    # The current approach with extending job_logs from local_job_logs inside download_file_to_path should work with ThreadPoolExecutor.
    logger.info(f"Job {job_id}: Starting download of {len(object_names)} files using ThreadPoolExecutor.")
    job_logs.append(f"INFO: Job {job_id}: Starting download of {len(object_names)} files using ThreadPoolExecutor.")
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(download_file_to_path, object_names))
        downloaded_files = [r for r in results if r is not None] # Filter out None results from failed downloads

    successful_downloads = len(downloaded_files)
    total_files = len(object_names)
    logger.info(f"Job {job_id}: Download process completed. Successfully downloaded {successful_downloads}/{total_files} files.")
    job_logs.append(f"INFO: Job {job_id}: Download process completed. Successfully downloaded {successful_downloads}/{total_files} files.")

    return downloaded_files, job_logs

def file_s3(bucket_name: str, object_name: str, job_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    '''
    Downloads a single file from a given S3 bucket and object name, file is given a random name.
    If the file is a zip file, it is extracted into a directory with the same name.

    Returns a tuple containing an object (with file_path, type, original_name, extracted_path) and a list of log messages.
    '''
    job_logs: List[str] = []
    log_prefix = f"Job {job_id}: " if job_id else ""

    try:
        os.makedirs('job_files', exist_ok=True)
        logger.info(f"{log_prefix}Created 'job_files' directory if it didn't exist.")
        job_logs.append(f"INFO: {log_prefix}Created 'job_files' directory if it didn't exist.")
    except OSError as e:
        logger.error(f"{log_prefix}Error creating 'job_files' directory: {e}")
        job_logs.append(f"ERROR: {log_prefix}Error creating 'job_files' directory: {e}")
        return None, job_logs

    s3_client, _ = get_boto_client(job_logs=job_logs) # Pass job_logs to capture boto client setup logs
    if not s3_client:
        logger.error(f"{log_prefix}Failed to get S3 client for file_s3 download. Aborting.")
        # job_logs already contains errors from get_boto_client
        return None, job_logs

    file_extension = os.path.splitext(object_name)[1].replace('.', '')
    file_name_uuid = f'{uuid.uuid4()}'
    output_file_path = os.path.join('job_files', f'{file_name_uuid}.{file_extension}')

    try:
        logger.info(f"{log_prefix}Attempting to download {bucket_name}/{object_name} to {output_file_path}")
        job_logs.append(f"INFO: {log_prefix}Attempting to download {bucket_name}/{object_name} to {output_file_path}")
        s3_client.download_file(bucket_name, object_name, output_file_path)
        logger.info(f"{log_prefix}File {object_name} downloaded successfully to {output_file_path}")
        job_logs.append(f"INFO: {log_prefix}File {object_name} downloaded successfully to {output_file_path}")
    except ClientError as e:
        logger.error(f"{log_prefix}ClientError downloading file {object_name}: {e}")
        job_logs.append(f"ERROR: {log_prefix}ClientError downloading file {object_name}: {e}")
        return None, job_logs
    except Exception as e:
        logger.error(f"{log_prefix}Unexpected error downloading file {object_name}: {e}")
        job_logs.append(f"ERROR: {log_prefix}Unexpected error downloading file {object_name}: {e}")
        return None, job_logs

    unziped_directory_abs_path: Optional[str] = None
    if file_extension == 'zip':
        unziped_directory_rel_path = os.path.join('job_files', file_name_uuid)
        try:
            os.makedirs(unziped_directory_rel_path, exist_ok=True)
            logger.info(f"{log_prefix}Created directory for unzipping: {unziped_directory_rel_path}")
            job_logs.append(f"INFO: {log_prefix}Created directory for unzipping: {unziped_directory_rel_path}")
            with zipfile.ZipFile(output_file_path, 'r') as zip_ref:
                zip_ref.extractall(unziped_directory_rel_path)
            unziped_directory_abs_path = os.path.abspath(unziped_directory_rel_path)
            logger.info(f"{log_prefix}File {output_file_path} unzipped successfully to {unziped_directory_abs_path}")
            job_logs.append(f"INFO: {log_prefix}File {output_file_path} unzipped successfully to {unziped_directory_abs_path}")
        except zipfile.BadZipFile as e:
            logger.error(f"{log_prefix}BadZipFile error when unzipping {output_file_path}: {e}")
            job_logs.append(f"ERROR: {log_prefix}BadZipFile error when unzipping {output_file_path}: {e}")
            # Decide if you want to return None or the downloaded zip path without extraction
            return None, job_logs # Or return the dict with extracted_path as None
        except Exception as e:
            logger.error(f"{log_prefix}Error unzipping file {output_file_path}: {e}")
            job_logs.append(f"ERROR: {log_prefix}Error unzipping file {output_file_path}: {e}")
            # Decide as above
            return None, job_logs

    result_data = {
        "file_path": os.path.abspath(output_file_path),
        "type": file_extension,
        "original_name": object_name,
        "extracted_path": unziped_directory_abs_path
    }
    return result_data, job_logs
