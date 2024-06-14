curl -X POST http://localhost:8000 \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "model": "tiny",
            "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
        }
    }'
curl -X POST http://localhost:8000 \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "model": "tiny",
            "bucket":"transcription",
            "audio": "test_1.m4a"
        }
    }'    