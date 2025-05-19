# docker login
docker build -f Dockerfile.core -t worker-faster_whisper_core:1.0 .
docker push fmatuszewski/worker-faster_whisper_core:1.0

docker build -f Dockerfile.child -t worker-faster_whisper:latest .
docker push fmatuszewski/worker-faster_whisper:latest

