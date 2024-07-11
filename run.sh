
docker build -t diffdock-fastapi -f Dockerfile_copy.diffdock-fastapi .
docker run -d -p 8000:8000 --gpus all diffdock-fastapi
