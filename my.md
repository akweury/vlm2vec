
### Building the ELVIS-C Docker Image
``` 
git clone https://github.com/akweury/vlm2vec.git
cd vlm2vec
docker build -t vlm2vec .
```

### Running the ELVIS-C Docker Container
``` 
docker run -it --gpus all -v /home/ml-jsha/ELVIS/grb:/grb --rm vlm2vec:latest
```

### Running ELVIS-C
```
python -m eval_elvis_c --batch_size 1 --principle continuity --device_id 4
```