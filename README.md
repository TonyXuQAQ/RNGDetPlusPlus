# [RAL 2023] RNGDet++
This is the official repo of paper **RNGDet++: Road Network Graph Detection by Transformer with Instance Segmentation and Multi-scale Features Enhancement** by Zhenhua Xu, Yuxuan Liu, Yuxiang Sun, Ming Liu and Lujia Wang.

<!-- **Note:** The implementation code of sampling and training will be released in a later stage. Only inference is open-sourced currently. -->

## Supplementary materials
For the demo video and supplementary document, please visit our [project page](https://tonyxuqaq.github.io/projects/RNGDetPlusPlus/).

## Update 
Mar/1/2023: Paper accepted by RA-L.

Dec/23/2022: Add SpaceNet dataset

Nov/28/2022: Release the initial version training code

Oct/23/2022: Update the Sat2Graph City-Scale dataset onto Google drive, since the raw data link provided by Sat2Graph is not valid any longer.

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
This repo is implemented in the docker container. **All experiments except evaluation are conducted within docker containers**. Make sure you have docker installed. Please refer to [install Docker](https://docs.docker.com/engine/install/ubuntu/) and [Docker beginner tutorial](https://docker-curriculum.com/) for more information.

### Docker image
```
cd docker
./build_image.bash
```
### Docker container
In ```./build_continer.bash```, set ```RNGDet_dir``` as the directory of this repo.
```
./build_continer_cityscale.bash # to try city scale dataset released by Sat2Graph
./build_continer_spacenet.bash # to try SpaceNet dataset 
```
**Note** We keep the raw code for the city scale dataset. For the new added spacenet dataset, we modify the processing stripts to better fit RNGDet++ to it, since the spacenet dataset has smaller images covering smaller regions, which has quite different characteristics with that of the city scale dataset. 


## Data preparation and pretrained checkpoints
Run the follow commands to prepare the dataset and pretrained checkpoints of RNGDet and RNGDet++.
```
cd prepare_dataset
./preprocessing.bash
```

### Update Oct/23/2022
The raw data download link provided by MIT is invalid now. We update the data to Google Drive.

### Update Dec/24/2022
The script to download the data from Google Drive is blocked. Please manually download the data and put it into ```prepare_dataset```. The Google Drive link could be found in the comment line in ```./prepare_dataset/preprocessing.bash```

### Update Feb/25/2026
Google Drive links have failed. Please download all data and checkpoints manually from https://cloud.tsinghua.edu.cn/d/d32cb7d4b19046ed9a42/.

## Sampling

Before training, run the sampler to generate traing samples:
```
./bash/run_sampler.sh
```
Parameters:
- ```edge_move_ahead_length``` (int, default=30): Max distance(pixels) moving ahead in each step.
- ```noise``` (int, default=8): Max random noise added during the sampling process (uniform distribution noise).
- ```max_num_frame``` (int, default=10000): Max number of samples generated for each large aerial image.

## Train

To train RNGDet, run 
```
./bash/run_train_RNGDet.sh
```

To train RNGDet++, run 
```
./bash/run_train_RNGDet++.sh
```
**Note**: Due to the randomness existing in both sampling and training, the final performance of the proposed models might be slightly different from the number reported in the paper. Please open an issue if you cannot produce the results.

## Inference
To try RNGDet, run 
```
./bash/run_test_RNGDet.sh
```

To try RNGDet++, run 
```
./bash/run_test_RNGDet++.sh
```

Parameters:
- ```candidate_filter_threshold``` (int, default=30): The distance threshold to filter initial candidated obtained from segmentation heatmap peaks. If one peak is too closed to the road network graph detected so far, it is filtered out.
- ```logit_threshold``` (float,0~1,default=0.75): The threshold to filter invalid vertices in the next step.
- ```extract_candidate_threshold``` (float,0~1,default=0.7): The threshold to detect local peaks in the segmentation heatmap to find initial candidates.
- ```alignment_distance``` (int, default=5): The distance threshold for graph merge and alignment. If a predicted vertex is too closed to predicted key vertices in the past, they are merged. 
- ```instance_seg``` (bool, default=False): Whether the instance segmentation head is used.
- ```multi_scale``` (bool, default=False): Whether multi-scale features are used.
- ```process_boundary``` (bool, default=False): Whether increase the logit_threshold near image boundaries.

Note: We provide the parameter setting in inference scripts of RNGDet and RNGDet++ in ```./bash``` that achieve the best performance.

## Evaluation
Go to ```{{ DATASET_NAME }}/metrics```. For TOPO metrics, run
```
./topo.bash
```

For APLS metrics, run
```
./apls.bash
```
Remember to set the path of predicted graphs in bash scripts.

**Note**: Evaluation metric scripts are not runnable in docker container. Please use them outside docker.

**Note**: Due to the randomness of RNGDet++ and evaluation metrics, the actual evaluation results might be slight different from the reported numbers in the paper.


## Contact
For any questions, please open an issue.

## Ackonwledgement
We thank the following open-sourced projects:

[SAT2GRAPH](https://github.com/songtaohe/Sat2Graph)

[DETR](https://github.com/facebookresearch/detr)

## Citation
```
@article{xu2023rngdet++,
  title={Rngdet++: Road network graph detection by transformer with instance segmentation and multi-scale features enhancement},
  author={Xu, Zhenhua and Liu, Yuxuan and Sun, Yuxiang and Liu, Ming and Wang, Lujia},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={5},
  pages={2991--2998},
  year={2023},
  publisher={IEEE}
}
@article{xu2022rngdet,
  title={Rngdet: Road network graph detection by transformer in aerial images},
  author={Xu, Zhenhua and Liu, Yuxuan and Gan, Lu and Sun, Yuxiang and Wu, Xinyu and Liu, Ming and Wang, Lujia},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--12},
  year={2022},
  publisher={IEEE}
}
@article{xu2021icurb,
  title={icurb: Imitation learning-based detection of road curbs using aerial images for autonomous driving},
  author={Xu, Zhenhua and Sun, Yuxiang and Liu, Ming},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={2},
  pages={1097--1104},
  year={2021},
  publisher={IEEE}
}
```

## License
GNU General Public License v3.0

Not allowed for commercial purposes. Only for academic research.
