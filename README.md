# RNGDet++
This is the official repo of paper **RNGDet++: Road Network Graph Detection by Transformer with Instance Segmentation and Multi-scale Features Enhancement** by Zhenhua Xu, Yuxuan Liu, Yuxiang Sun, Ming Liu and Lujia Wang.

# Supplementary materials
For the demo video and supplementary document, please visit our [project page](https://tonyxuqaq.github.io/projects/RNGDetPlusPlus/).

## Update 
Sep/21/2022: Release the inference code

## Platform info
Hardware:
```
GPU: 4 RTX3090
CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
RAM: 256G
SSD: 4T
```
Software:
```
Ubuntu 20.04.3 LTS
CUDA 11.1
Docker 20.10.7
Nvidia-driver 495.29.05
```
## Docker 
This repo is implemented in the docker container. Make sure you have docker installed. Please refer to [install Docker](https://docs.docker.com/engine/install/ubuntu/) and [Docker beginner tutorial](https://docker-curriculum.com/) for more information.

### Docker image
```
cd docker
./build_image.bash
```
### Docker container
In ```./build_continer.bash```, set ```RNGDet_dir``` as the directory of this repo.
```
./build_continer.bash
```

## Data preparation and pretrained checkpoints
Run the follow commands to prepare the dataset and pretrained checkpoints of RNGDet and RNGDet++.
```
cd prepare_dataset
./preprocessing.bash
```

### Update Oct/23/2022
The raw data download link provided by MIT is invalid now. We update the data to Google Drive.


## Train
Comming soon...

## Inference
To try RNGDet, run 
```
./bash/run_test_RNGDet.bash
```

To try RNGDet++, run 
```
./bash/run_test_RNGDet++.bash
```

Parameters:
- ```candidate_filter_threshold``` (int, default=30): The distance threshold to filter initial candidated obtained from segmentation heatmap peaks. If one peak is too closed to the road network graph detected so far, it is filtered out.
- ```logit_threshold``` (float,0~1,default=0.75): The threshold to filter invalid vertices in the next step.
- ```extract_candidate_threshold``` (floar,0~1,default=0.7): The threshold to detect local peaks in the segmentation heatmap to find initial candidates.
- ```alignment_distance``` (int, default=5): The distance threshold for graph merge and alignment. If a predicted vertex is too closed to predicted key vertices in the past, they are merged. 
- ```instance_seg``` (bool, default=False): Whether the instance segmentation head is used.
- ```multi_scale``` (bool, default=False): Whether multi-scale features are used.
- ```process_boundary``` (bool, default=False): Whether increase the logit_threshold near image boundaries.

Note: We provide the parameter setting in inference scripts of RNGDet and RNGDet++ in ```./bash``` that achieve the best performance.

## Evaluation
Go to ```metrics```. For TOPO metrics, run
```
./topo.bash
```

For APLS metrics, run
```
./apls.bash
```
Remember to set the path of predicted graphs in bash scripted.

**Note**: Evaluation metric scripts are not runnable in docker container. Please use them outside docker.


## Contact
For any questions, please open an issue.

## Ackonwledgement
We thank the following open-sourced projects:

[SAT2GRAPH](https://github.com/songtaohe/Sat2Graph)

[DETR](https://github.com/facebookresearch/detr)

## Citation


## License
GNU General Public License v3.0