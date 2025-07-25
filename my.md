
### Building the ELVIS-C Docker Image
``` 
git clone https://github.com/akweury/vlm2vec.git
cd vlm2vec
docker build -t vlm2vec .
```

### Running the ELVIS-C Docker Container
``` 
docker run -it --gpus all --rm vlm2vec:latest
```