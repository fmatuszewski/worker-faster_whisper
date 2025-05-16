# We need following env variables to run server
./env.sh

# Run local server
python rp_handler.py \
  --rp_serve_api \
  --rp_api_port 8000 \
  --rp_api_concurrency 4


# Basic test download publicly available file
curl -X POST http://localhost:8000/runsync \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "model": "medium",
            "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
            }
        }'

# Test if we can transcribe
curl -X POST http://localhost:8000/runsync \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "model": "medium",
            "language": "pl"
            "bucket":"audio",
            "audio": "test_1.m4a"
        }
    }'

curl -X POST http://localhost:8000/runsync \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "model": "medium",
            "language": "pl"
            "bucket":"audio",
            "audio": "audio_1732101620318.m4a"
        }
    }'



curl -X POST http://localhost:8000/runsync \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d '{
          "input": {
             "no_speech_threshold": 0.1,
              "language": "pl",
              "model": "medium",
              "bucket": "audio",
              "audio": "f02e51c7-140b-4824-971d-532f65307385/1f8f1f06-6ba8-4973-8d2b-2939168a9d9b/audio_1731435800759.m4a"
          }
        }'





