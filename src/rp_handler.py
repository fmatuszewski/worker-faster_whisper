"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import tempfile
import logging
import os
from typing import List, Dict, Any, Tuple, Optional

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from rp_download import download_files_from_s3 # Modified to return logs
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL: Optional[predict.Predictor] = None # Initialize as None

def initialize_model(job_logs: List[str]) -> bool:
    global MODEL
    if MODEL is None:
        try:
            logger.info("Initializing model...")
            job_logs.append("INFO: Initializing model...")
            MODEL = predict.Predictor()
            MODEL.setup(job_logs) # Pass job_logs to setup
            logger.info("Model initialized successfully.")
            job_logs.append("INFO: Model initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Fatal error during model initialization: {e}", exc_info=True)
            job_logs.append(f"FATAL: Error during model initialization: {e}")
            MODEL = None # Ensure model is None if setup fails
            return False
    return True


logger.info(f"BUCKET_ENDPOINT_URL: {os.environ.get('BUCKET_ENDPOINT_URL', None)}")
logger.info(f"BUCKET_ACCESS_KEY_ID: {os.environ.get('BUCKET_ACCESS_KEY_ID', None)}")
logger.info(f"BUCKET_REGION: {os.getenv('BUCKET_REGION')}")


def base64_to_tempfile(base64_file: str, job_logs: List[str]) -> Optional[str]:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file
    job_logs (List[str]): List to store log messages.

    Returns:
    Optional[str]: Path to tempfile or None if an error occurs.
    '''
    try:
        logger.info("Decoding base64 string to temporary file.")
        job_logs.append("INFO: Decoding base64 string to temporary file.")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(base64.b64decode(base64_file))
        logger.info(f"Base64 decoded and saved to temporary file: {temp_file.name}")
        job_logs.append(f"INFO: Base64 decoded and saved to temporary file: {temp_file.name}")
        return temp_file.name
    except base64.binascii.Error as e:
        logger.error(f"Error decoding base64 string: {e}", exc_info=True)
        job_logs.append(f"ERROR: Error decoding base64 string: {e}")
        return None
    except IOError as e:
        logger.error(f"IOError creating or writing to temporary file: {e}", exc_info=True)
        job_logs.append(f"ERROR: IOError creating or writing to temporary file: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in base64_to_tempfile: {e}", exc_info=True)
        job_logs.append(f"ERROR: Unexpected error in base64_to_tempfile: {e}")
        return None


@rp_debugger.FunctionTimer
def run_whisper_job(job: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction or an error object with detailed_logs
    '''
    job_id = job.get('id', 'unknown_job')
    job_logs: List[str] = [f"INFO: Starting job ID: {job_id}"]
    logger.info(f"Starting job ID: {job_id}")

    global MODEL
    if MODEL is None:
        if not initialize_model(job_logs):
            logger.error(f"Job {job_id}: Model initialization failed. Cannot proceed.")
            job_logs.append(f"ERROR: Job {job_id}: Model initialization failed. Cannot proceed.")
            return {"error": "Job failed due to model initialization error", "detailed_logs": job_logs}
    
    job_input = job.get('input', {})
    if not job_input:
        logger.error(f"Job {job_id}: No 'input' found in job payload.")
        job_logs.append(f"ERROR: Job {job_id}: No 'input' found in job payload.")
        return {"error": "Missing 'input' in job payload", "detailed_logs": job_logs}

    # Input Validation
    try:
        logger.info(f"Job {job_id}: Starting input validation.")
        job_logs.append(f"INFO: Job {job_id}: Starting input validation.")
        with rp_debugger.LineTimer('validation_step'):
            input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            error_msg = f"Job {job_id}: Input validation failed: {input_validation['errors']}"
            logger.error(error_msg)
            job_logs.append(f"ERROR: {error_msg}")
            return {"error": "Input validation failed", "detailed_errors": input_validation['errors'], "detailed_logs": job_logs}
        job_input = input_validation['validated_input']
        logger.info(f"Job {job_id}: Input validation successful.")
        job_logs.append(f"INFO: Job {job_id}: Input validation successful.")
    except Exception as e:
        error_msg = f"Job {job_id}: Unexpected error during input validation: {e}"
        logger.error(error_msg, exc_info=True)
        job_logs.append(f"ERROR: {error_msg}")
        return {"error": "Job failed during input validation", "detailed_logs": job_logs}

    # Audio source validation
    has_audio_url = job_input.get('audio', False)
    has_audio_base64 = job_input.get('audio_base64', False)

    if not has_audio_url and not has_audio_base64:
        error_msg = f"Job {job_id}: Must provide either 'audio' URL or 'audio_base64'."
        logger.error(error_msg)
        job_logs.append(f"ERROR: {error_msg}")
        return {'error': error_msg, "detailed_logs": job_logs}

    if has_audio_url and has_audio_base64:
        error_msg = f"Job {job_id}: Must provide either 'audio' URL or 'audio_base64', not both."
        logger.error(error_msg)
        job_logs.append(f"ERROR: {error_msg}")
        return {'error': error_msg, "detailed_logs": job_logs}

    audio_input_path: Optional[str] = None

    # Download audio or decode base64
    try:
        if has_audio_url:
            is_s3_bucket = job_input.get('bucket', False)
            audio_url = job_input['audio']
            logger.info(f"Job {job_id}: Audio source is URL: {audio_url}, S3: {is_s3_bucket}")
            job_logs.append(f"INFO: Job {job_id}: Audio source is URL: {audio_url}, S3: {is_s3_bucket}")
            with rp_debugger.LineTimer('download_step'):
                if is_s3_bucket:
                    bucket_name = job_input['bucket']
                    logger.info(f"Job {job_id}: Downloading from S3 bucket '{bucket_name}', object '{audio_url}'.")
                    job_logs.append(f"INFO: Job {job_id}: Downloading from S3 bucket '{bucket_name}', object '{audio_url}'.")
                    downloaded_files, s3_logs = download_files_from_s3(job_id, bucket_name, [audio_url], job_input.get('bucket_creds'))
                    job_logs.extend(s3_logs)
                    if downloaded_files and downloaded_files[0]:
                        audio_input_path = downloaded_files[0]
                        logger.info(f"Job {job_id}: Successfully downloaded from S3 to {audio_input_path}.")
                        job_logs.append(f"INFO: Job {job_id}: Successfully downloaded from S3 to {audio_input_path}.")
                    else:
                        error_msg = f"Job {job_id}: Failed to download audio from S3. Bucket: {bucket_name}, Object: {audio_url}."
                        logger.error(error_msg)
                        job_logs.append(f"ERROR: {error_msg}")
                        return {'error': 'Failed to download audio from S3', "detailed_logs": job_logs}
                else:
                    logger.info(f"Job {job_id}: Downloading from public URL: {audio_url}.")
                    job_logs.append(f"INFO: Job {job_id}: Downloading from public URL: {audio_url}.")
                    # download_files_from_urls doesn't return logs, so we wrap and log manually
                    try:
                        downloaded_paths = download_files_from_urls(job_id, [audio_url])
                        if downloaded_paths and downloaded_paths[0]:
                            audio_input_path = downloaded_paths[0]
                            logger.info(f"Job {job_id}: Successfully downloaded from URL to {audio_input_path}.")
                            job_logs.append(f"INFO: Job {job_id}: Successfully downloaded from URL to {audio_input_path}.")
                        else:
                            raise Exception("download_files_from_urls returned empty or invalid result.")
                    except Exception as e_url_download:
                        error_msg = f"Job {job_id}: Failed to download audio from URL {audio_url}: {e_url_download}"
                        logger.error(error_msg, exc_info=True)
                        job_logs.append(f"ERROR: {error_msg}")
                        return {'error': 'Failed to download audio from URL', "detailed_logs": job_logs}

        elif has_audio_base64:
            logger.info(f"Job {job_id}: Audio source is base64 encoded string.")
            job_logs.append(f"INFO: Job {job_id}: Audio source is base64 encoded string.")
            audio_input_path = base64_to_tempfile(job_input['audio_base64'], job_logs)
            if not audio_input_path:
                error_msg = f"Job {job_id}: Failed to process base64 audio."
                logger.error(error_msg)
                # job_logs already contains details from base64_to_tempfile
                return {'error': 'Failed to process base64 audio', "detailed_logs": job_logs}
            logger.info(f"Job {job_id}: Successfully processed base64 audio to {audio_input_path}.")
            job_logs.append(f"INFO: Job {job_id}: Successfully processed base64 audio to {audio_input_path}.")

    except Exception as e:
        error_msg = f"Job {job_id}: Error during audio file processing/download: {e}"
        logger.error(error_msg, exc_info=True)
        job_logs.append(f"ERROR: {error_msg}")
        return {"error": "Job failed during audio processing", "detailed_logs": job_logs}

    if not audio_input_path: # Should be caught by earlier checks, but as a safeguard
        error_msg = f"Job {job_id}: Audio input path was not set after download/decode phase."
        logger.error(error_msg)
        job_logs.append(f"ERROR: {error_msg}")
        return {"error": "Critical error: Audio input not available for prediction", "detailed_logs": job_logs}

    # Prediction
    try:
        logger.info(f"Job {job_id}: Starting prediction with audio: {audio_input_path}")
        job_logs.append(f"INFO: Job {job_id}: Starting prediction with audio: {audio_input_path}")
        with rp_debugger.LineTimer('prediction_step'):
            # Ensure MODEL is not None (should be handled by initialization)
            if MODEL is None:
                 job_logs.append(f"CRITICAL: Job {job_id}: MODEL is None before prediction call.")
                 logger.critical(f"Job {job_id}: MODEL is None before prediction call. This should not happen.")
                 return {"error": "Model not available for prediction", "detailed_logs": job_logs}

            whisper_results, predict_logs = MODEL.predict(
                audio=audio_input_path,
                model_name=os.getenv("MODEL_SIZE", "tiny"),
                transcription=job_input["transcription"],
                translate=job_input["translate"],
                language=job_input["language"],
                temperature=job_input["temperature"],
                best_of=job_input["best_of"],
                beam_size=job_input["beam_size"],
                patience=job_input["patience"],
                length_penalty=job_input["length_penalty"],
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input["initial_prompt"],
                condition_on_previous_text=job_input["condition_on_previous_text"],
                temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                compression_ratio_threshold=job_input["compression_ratio_threshold"],
                logprob_threshold=job_input["logprob_threshold"],
                no_speech_threshold=job_input["no_speech_threshold"],
                enable_vad=job_input["enable_vad"],
                word_timestamps=job_input["word_timestamps"],
                job_id=job_id # Pass job_id for logging within predict
            )
            job_logs.extend(predict_logs)
        logger.info(f"Job {job_id}: Prediction successful.")
        job_logs.append(f"INFO: Job {job_id}: Prediction successful.")
    except ValueError as ve: # Catch specific errors like model not found from predict
        error_msg = f"Job {job_id}: ValueError during prediction: {ve}"
        logger.error(error_msg, exc_info=True)
        job_logs.append(f"ERROR: {error_msg}")
        return {"error": f"Job failed during prediction: {ve}", "detailed_logs": job_logs}
    except Exception as e:
        error_msg = f"Job {job_id}: Unexpected error during prediction: {e}"
        logger.error(error_msg, exc_info=True)
        job_logs.append(f"ERROR: {error_msg}")
        return {"error": "Job failed during prediction", "detailed_logs": job_logs}

    # Cleanup
    try:
        logger.info(f"Job {job_id}: Starting cleanup.")
        job_logs.append(f"INFO: Job {job_id}: Starting cleanup.")
        with rp_debugger.LineTimer('cleanup_step'):
            # rp_cleanup.clean(['input_objects']) # This might delete the temp file if it's in 'input_objects'
            # Let's be more specific or ensure temp files are handled correctly.
            # If audio_input_path was a temp file, it should be cleaned.
            # The 'jobs/{job_id}/downloaded_files' directory can also be cleaned.
            paths_to_clean = [os.path.dirname(audio_input_path)] if audio_input_path and 'temp' not in audio_input_path.lower() else []
            if audio_input_path and 'temp' in audio_input_path.lower() and os.path.exists(audio_input_path):
                paths_to_clean.append(audio_input_path)
            
            # Clean the 'jobs/{job_id}' directory structure
            job_download_dir = os.path.abspath(os.path.join('jobs', job_id))
            if os.path.exists(job_download_dir):
                 paths_to_clean.append(job_download_dir)

            if paths_to_clean:
                rp_cleanup.clean(paths_to_clean)
                logger.info(f"Job {job_id}: Cleanup successful for paths: {paths_to_clean}.")
                job_logs.append(f"INFO: Job {job_id}: Cleanup successful for paths: {paths_to_clean}.")
            else:
                logger.info(f"Job {job_id}: No specific paths identified for cleanup beyond default rp_cleanup behavior.")
                job_logs.append(f"INFO: Job {job_id}: No specific paths identified for cleanup.")

    except Exception as e:
        error_msg = f"Job {job_id}: Error during cleanup: {e}"
        logger.warning(error_msg, exc_info=True) # Warning as cleanup failure might not be critical
        job_logs.append(f"WARNING: {error_msg}")

    logger.info(f"Job {job_id}: Processing completed successfully.")
    job_logs.append(f"INFO: Job {job_id}: Processing completed successfully.")
    
    # Add job_logs to the successful result if needed for audit, or just return whisper_results
    # For now, only return logs on error as per requirements.
    # whisper_results['job_logs'] = job_logs # Optionally include logs in success
    return whisper_results


# Initialize model once at startup, outside handler if possible, or lazily.
# For serverless, lazy initialization within the first call to handler is common.
# Global MODEL variable will be set by initialize_model on first job.

runpod.serverless.start({"handler": run_whisper_job})
