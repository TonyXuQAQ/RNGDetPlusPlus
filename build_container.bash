#!/bin/bash

CMD=$*

if [ -z "$CMD"];
then 
	CMD=/bin/bash
fi

# Set the directory here
RNGDet_dir=/home/tonyx/final_repo/InstanceRNGDet

home_dir=$RNGDet_dir
dataset_dir=$RNGDet_dir/data
container_name=RNGDetplusplus
port_number=5050

docker run \
	-v $home_dir:/tonyxu\
	-v $dataset_dir:/tonyxu/dataset\
	--name=$container_name\
	--gpus all\
	--shm-size 32G\
	-p $port_number:6006\
	--rm -it zhxu_1.8.0-cuda11.1-cudnn8_py3.8 $CMD

docker attach RNGDetplusplus