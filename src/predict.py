"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp
from pathlib import Path

# Configure logging
# Using the same logger instance as rp_handler if this runs in the same process,
# or a new one if it's separate. For consistency, configure it.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Predictor:
    """ A Predictor class for the Whisper model """

    def __init__(self):
        self.models: Dict[str, WhisperModel] = {}
        logger.info("Predictor initialized. Models dictionary created.")

    def load_model(self, model_name: str, job_logs: List[str]) -> Tuple[Optional[str], Optional[WhisperModel]]:
        """ Load the model from the weights folder. """
        device = "cuda" if rp_cuda.is_available() else "cpu"
        compute_type = "float16" if rp_cuda.is_available() else "int8"
        log_msg = f"Attempting to load model: {model_name} on device: {device} with compute_type: {compute_type}"
        logger.info(log_msg)
        job_logs.append(f"INFO: {log_msg}")
        try:
            loaded_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type
            )
            success_msg = f"Model {model_name} loaded successfully."
            logger.info(success_msg)
            job_logs.append(f"INFO: {success_msg}")
            return model_name, loaded_model
        except Exception as e:
            error_msg = f"Error loading model {model_name}: {e}"
            logger.error(error_msg, exc_info=True)
            job_logs.append(f"ERROR: {error_msg}")
            return None, None

    def setup(self, job_logs: List[str]):
        """Load the model into memory to make running multiple predictions efficient"""
        # model_names = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
        model_names_to_load = ["tiny", "base", "medium"] # large-v3 can be very resource intensive
        logger.info(f"Starting model setup. Attempting to load models: {model_names_to_load}")
        job_logs.append(f"INFO: Starting model setup. Attempting to load models: {model_names_to_load}")

        # Sequentially load models to better manage memory and provide clearer logs if one fails
        loaded_count = 0
        for model_name_to_load in model_names_to_load:
            if model_name_to_load not in self.models: # Avoid reloading if already present
                name, model = self.load_model(model_name_to_load, job_logs)
                if name and model:
                    self.models[name] = model
                    loaded_count +=1
                else:
                    # Log already contains error details from load_model
                    logger.warning(f"Skipping further model loading due to failure with {model_name_to_load}.")
                    job_logs.append(f"WARNING: Skipping further model loading due to failure with {model_name_to_load}.")
                    # Depending on requirements, you might want to raise an error here if a critical model fails
                    # For now, we'll allow partial loading.
        
        if loaded_count == len(model_names_to_load):
            logger.info(f"All {len(model_names_to_load)} specified models loaded successfully.")
            job_logs.append(f"INFO: All {len(model_names_to_load)} specified models loaded successfully.")
        else:
            logger.warning(f"Model setup completed. Successfully loaded {loaded_count}/{len(model_names_to_load)} models. Check logs for errors.")
            job_logs.append(f"WARNING: Model setup completed. Successfully loaded {loaded_count}/{len(model_names_to_load)} models. Check logs for errors.")


    def predict(
        self,
        audio: str,
        model_name: str = "base",
        transcription: str = "plain_text",
        translate: bool = False,
        language: Optional[str] = None,
        temperature: float = 0.0, # Changed to float
        best_of: int = 5,
        beam_size: int = 5,
        patience: float = 1.0, # Changed to float
        length_penalty: Optional[float] = None, # Changed to float
        suppress_tokens: str = "-1",
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        temperature_increment_on_fallback: Optional[float] = 0.2, # Changed to float
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        enable_vad: bool = False,
        word_timestamps: bool = False,
        job_id: Optional[str] = "N/A" # Added for logging context
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Run a single prediction on the model.
        Returns a tuple: (results_dict, logs_list)
        """
        predict_logs: List[str] = []
        log_prefix = f"Job {job_id} - Predictor: "

        logger.info(f"{log_prefix}Received prediction request for model: {model_name}, audio: {audio}")
        predict_logs.append(f"INFO: {log_prefix}Received prediction request for model: {model_name}, audio: {audio}")

        model = self.models.get(model_name)
        if not model:
            error_msg = f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
            logger.error(f"{log_prefix}{error_msg}")
            predict_logs.append(f"ERROR: {log_prefix}{error_msg}")
            # Instead of raising ValueError directly, return it in a way rp_handler can process
            # This allows rp_handler to append these logs to its main job_logs
            raise ValueError(error_msg) # rp_handler will catch this

        actual_temperature: Union[List[float], Tuple[float, ...]]
        if temperature_increment_on_fallback is not None:
            actual_temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
            logger.info(f"{log_prefix}Using temperature fallback range: {actual_temperature}")
            predict_logs.append(f"INFO: {log_prefix}Using temperature fallback range: {actual_temperature}")
        else:
            actual_temperature = [temperature]
            logger.info(f"{log_prefix}Using fixed temperature: {actual_temperature}")
            predict_logs.append(f"INFO: {log_prefix}Using fixed temperature: {actual_temperature}")
        
        # Parse suppress_tokens
        try:
            parsed_suppress_tokens: Optional[List[int]] = [int(token.strip()) for token in suppress_tokens.split(",")] if suppress_tokens and suppress_tokens != "-1" else [-1]
        except ValueError:
            warn_msg = f"{log_prefix}Invalid suppress_tokens format: '{suppress_tokens}'. Defaulting to [-1]."
            logger.warning(warn_msg)
            predict_logs.append(f"WARNING: {warn_msg}")
            parsed_suppress_tokens = [-1]


        try:
            logger.info(f"{log_prefix}Starting transcription for audio: {audio} with model {model_name}")
            predict_logs.append(f"INFO: {log_prefix}Starting transcription for audio: {audio} with model {model_name}")
            
            audio_path = Path(audio).expanduser().resolve()

            if not audio_path.exists():
                error_msg = f"[Predictor] Audio file missing: {audio_path}"
                logger.error(error_msg)
                predict_logs.append(f"ERROR: {log_prefix}{error_msg}")
                raise FileNotFoundError(error_msg)  
             
            segments_iterable, info = model.transcribe(
                audio=str(audio), # Ensure audio is string path
                language=language,
                task="transcribe", # "transcribe" or "translate"
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=actual_temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                prefix=None, # Not typically user-set
                suppress_blank=True, # Default from faster-whisper
                suppress_tokens=parsed_suppress_tokens,
                without_timestamps=False, # We need timestamps for SRT/VTT and segments
                max_initial_timestamp=1.0, # Default
                word_timestamps=word_timestamps,
                vad_filter=enable_vad
            )

            # Convert generator to list to process multiple times
            segments = list(segments_iterable)
            
            logger.info(f"{log_prefix}Transcription completed. Detected language: {info.language}, Duration: {info.duration}s")
            predict_logs.append(f"INFO: {log_prefix}Transcription completed. Detected language: {info.language}, Duration: {info.duration}s, Segments found: {len(segments)}")

        except Exception as e:
            error_msg = f"Error during model.transcribe call: {e}"
            logger.error(f"{log_prefix}{error_msg}", exc_info=True)
            predict_logs.append(f"ERROR: {log_prefix}{error_msg}")
            # Re-raise to be caught by rp_handler
            raise RuntimeError(f"Transcription failed: {e}")


        output_transcription: str
        if transcription == "plain_text":
            output_transcription = " ".join([segment.text.lstrip() for segment in segments])
        elif transcription == "formatted_text":
            output_transcription = "\n".join([segment.text.lstrip() for segment in segments])
        elif transcription == "srt":
            output_transcription = write_srt(segments, predict_logs, log_prefix)
        elif transcription == "vtt": # Added vtt as an explicit option
            output_transcription = write_vtt(segments, predict_logs, log_prefix)
        else: # Default to plain text if unknown format
            warn_msg = f"Unknown transcription format '{transcription}'. Defaulting to 'plain_text'."
            logger.warning(f"{log_prefix}{warn_msg}")
            predict_logs.append(f"WARNING: {log_prefix}{warn_msg}")
            output_transcription = " ".join([segment.text.lstrip() for segment in segments])
        
        logger.info(f"{log_prefix}Formatted transcription (format: {transcription}). Length: {len(output_transcription)} chars.")
        predict_logs.append(f"INFO: {log_prefix}Formatted transcription (format: {transcription}).")

        translation_output: Optional[str] = None
        if translate:
            try:
                logger.info(f"{log_prefix}Starting translation task for audio: {audio}")
                predict_logs.append(f"INFO: {log_prefix}Starting translation task for audio: {audio}")
                translation_segments_iterable, translation_info = model.transcribe(
                    str(audio),
                    task="translate",
                    beam_size=beam_size, # Consider if translate needs different beam/temp settings
                    temperature=actual_temperature
                    # Other relevant parameters for translate task if needed
                )
                translation_segments = list(translation_segments_iterable)
                translation_output = write_srt(translation_segments, predict_logs, log_prefix + "[Translation]") # Assuming SRT for translation
                logger.info(f"{log_prefix}Translation completed. Detected language: {translation_info.language}, Duration: {translation_info.duration}s")
                predict_logs.append(f"INFO: {log_prefix}Translation completed. Detected language: {translation_info.language}, Segments: {len(translation_segments)}")
            except Exception as e_translate:
                error_msg = f"Error during translation task: {e_translate}"
                logger.error(f"{log_prefix}{error_msg}", exc_info=True)
                predict_logs.append(f"ERROR: {log_prefix}{error_msg}")
                # Continue without translation if it fails, or raise error based on requirements

        results = {
            "segments": format_segments(segments, predict_logs, log_prefix),
            "detected_language": info.language,
            "transcription": output_transcription,
            "translation": translation_output,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model_name": model_name, # Changed from "model" to "model_name" for clarity
            "audio_duration_seconds": info.duration,
        }

        if word_timestamps:
            logger.info(f"{log_prefix}Extracting word timestamps.")
            predict_logs.append(f"INFO: {log_prefix}Extracting word timestamps.")
            extracted_word_timestamps = []
            for segment_idx, segment in enumerate(segments):
                if segment.words:
                    for word_idx, word in enumerate(segment.words):
                        extracted_word_timestamps.append({
                            "word": word.word,
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "probability": round(word.probability, 3) if hasattr(word, 'probability') else None,
                            "segment_id": segment.id,
                            # "word_id_in_segment": word_idx # If useful
                        })
                else: # Log if a segment has no words, which might be unexpected for word_timestamps=True
                    no_words_msg = f"Segment {segment.id} (idx {segment_idx}) has no word timestamps."
                    logger.debug(f"{log_prefix}{no_words_msg}") # Debug as it might be normal for empty segments
                    predict_logs.append(f"DEBUG: {log_prefix}{no_words_msg}")

            results["word_timestamps"] = extracted_word_timestamps
            logger.info(f"{log_prefix}Extracted {len(extracted_word_timestamps)} word timestamps.")
            predict_logs.append(f"INFO: {log_prefix}Extracted {len(extracted_word_timestamps)} word timestamps.")
        
        logger.info(f"{log_prefix}Prediction process finished successfully.")
        predict_logs.append(f"INFO: {log_prefix}Prediction process finished successfully.")
        return results, predict_logs


def format_segments(transcript: List[Any], job_logs: List[str], log_prefix: str) -> List[Dict[str, Any]]:
    '''
    Format the segments to be returned in the API response.
    Includes logging for the formatting process.
    '''
    logger.debug(f"{log_prefix}Formatting {len(transcript)} segments.")
    job_logs.append(f"DEBUG: {log_prefix}Formatting {len(transcript)} segments.")
    formatted_segments = []
    try:
        for i, segment in enumerate(transcript):
            # Ensure all expected attributes exist, providing defaults or logging if not
            segment_data = {
                "id": getattr(segment, 'id', i), # Use index as fallback ID
                "seek": round(getattr(segment, 'seek', 0.0), 3),
                "start": round(getattr(segment, 'start', 0.0), 3),
                "end": round(getattr(segment, 'end', 0.0), 3),
                "text": getattr(segment, 'text', ""),
                "tokens": getattr(segment, 'tokens', []),
                "temperature": round(getattr(segment, 'temperature', 0.0), 2),
                "avg_logprob": round(getattr(segment, 'avg_logprob', 0.0), 4),
                "compression_ratio": round(getattr(segment, 'compression_ratio', 0.0), 2),
                "no_speech_prob": round(getattr(segment, 'no_speech_prob', 0.0), 4)
            }
            formatted_segments.append(segment_data)
        logger.debug(f"{log_prefix}Successfully formatted {len(formatted_segments)} segments.")
        job_logs.append(f"DEBUG: {log_prefix}Successfully formatted {len(formatted_segments)} segments.")
    except AttributeError as ae:
        error_msg = f"AttributeError while formatting segments: {ae}. Segment data might be incomplete."
        logger.error(f"{log_prefix}{error_msg}", exc_info=True)
        job_logs.append(f"ERROR: {log_prefix}{error_msg}")
        # Continue with what has been formatted, or decide to return empty list / raise error
    except Exception as e:
        error_msg = f"Unexpected error while formatting segments: {e}."
        logger.error(f"{log_prefix}{error_msg}", exc_info=True)
        job_logs.append(f"ERROR: {log_prefix}{error_msg}")
    return formatted_segments


def write_vtt(transcript: List[Any], job_logs: List[str], log_prefix: str) -> str:
    '''
    Write the transcript in VTT format.
    Includes logging.
    '''
    logger.info(f"{log_prefix}Writing VTT for {len(transcript)} segments.")
    job_logs.append(f"INFO: {log_prefix}Writing VTT for {len(transcript)} segments.")
    result = "WEBVTT\n\n" # Standard VTT header

    try:
        for segment in transcript:
            start_time = format_timestamp(getattr(segment, 'start', 0.0))
            end_time = format_timestamp(getattr(segment, 'end', 0.0))
            text = getattr(segment, 'text', "").strip().replace('-->', '->')
            result += f"{start_time} --> {end_time}\n"
            result += f"{text}\n\n"
        logger.info(f"{log_prefix}VTT content generated successfully.")
        job_logs.append(f"INFO: {log_prefix}VTT content generated successfully.")
    except Exception as e:
        error_msg = f"Error generating VTT content: {e}"
        logger.error(f"{log_prefix}{error_msg}", exc_info=True)
        job_logs.append(f"ERROR: {log_prefix}{error_msg}")
        # Return partial result or empty string depending on desired error handling
    return result


def write_srt(transcript: List[Any], job_logs: List[str], log_prefix: str) -> str:
    '''
    Write the transcript in SRT format.
    Includes logging.
    '''
    logger.info(f"{log_prefix}Writing SRT for {len(transcript)} segments.")
    job_logs.append(f"INFO: {log_prefix}Writing SRT for {len(transcript)} segments.")
    result = ""

    try:
        for i, segment in enumerate(transcript, start=1):
            start_time = format_timestamp(getattr(segment, 'start', 0.0), always_include_hours=True, decimal_marker=',')
            end_time = format_timestamp(getattr(segment, 'end', 0.0), always_include_hours=True, decimal_marker=',')
            text = getattr(segment, 'text', "").strip().replace('-->', '->')
            result += f"{i}\n"
            result += f"{start_time} --> {end_time}\n"
            result += f"{text}\n\n"
        logger.info(f"{log_prefix}SRT content generated successfully.")
        job_logs.append(f"INFO: {log_prefix}SRT content generated successfully.")
    except Exception as e:
        error_msg = f"Error generating SRT content: {e}"
        logger.error(f"{log_prefix}{error_msg}", exc_info=True)
        job_logs.append(f"ERROR: {log_prefix}{error_msg}")
    return result
