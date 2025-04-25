docker rmi $(docker images -qa -f 'dangling=true')d
docker build . -t land-certificate-ocr:0.1-gpu
docker run -it --rm --gpus all -v $(pwd)/data:/app/data -p 8501:8501 --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=1g  land-certificate-ocr:0.1-gpu
# docker run -it --rm -v $(pwd)/data:/app/data -p 8501:8501  land-certificate-ocr:0.1-gpu


