

curl -X POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d '{
        "input": {
            "model": "medium",
            "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
            }
        }'

curl -X POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "model": "medium",
            "bucket":"transcription",
            "audio": "test_1.m4a"
        }
    }'






