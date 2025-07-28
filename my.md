
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



### File Structure
```
grb/
├── symmetry/
│   ├── train/
│   │   ├── task_001/
│   │   │   ├── positive/
│   │   │   │   ├── 00000.png
│   │   │   │   ├── 00000.json
│   │   │   │   └── ... (up to 00009)
│   │   │   ├── negative/
│   │   │   │   ├── 00000.png
│   │   │   │   ├── 00000.json
│   │   │   │   └── ... (up to 00009)
│   │   └── ... (more task folders)
│   ├── test/
│   │   └── ... (same structure as train)
├── similarity/
│   └── ... (same as symmetry)
├── proximity/
│   └── ... (same as symmetry)
├── continuity/
│   └── ... (same as symmetry)
├── closure/
│   └── ... (same as symmetry)
```
