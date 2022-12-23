# source directory
sudo python3 spacenet_convert.py
dir=RNGDet_multi_ins/test

sudo python ./topo/main.py -savedir $dir
sudo python3 topo.py -savedir $dir