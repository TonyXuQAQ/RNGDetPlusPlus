# Download spacenet3 dataset
# if you cannot download this by script, download it manually at https://drive.google.com/file/d/1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W/view?usp=share_link
gdown https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W
unzip RGB_1.0_meter_full.zip
# rm -rf RGB_1.0_meter_full.zip 
mkdir -p ../data
mv ./RGB_1.0_meter ../data
mv ../data/RGB_1.0_meter/dataset.json ../data/data_split.json

# Generate label
echo "Generating labels ..."
python3 create_label.py
# echo "Finsh generating labels!"

# Get pretrained checkpoints
# if you cannot download this by script, download it manually at https://drive.google.com/file/d/1RAhy-CzRfYURwDPDbBdfksRxCttaJTQV/view?usp=share_link
gdown https://drive.google.com/uc?id=1RAhy-CzRfYURwDPDbBdfksRxCttaJTQV
unzip pretrain_spacenet.zip
rm -rf pretrain_spacenet.zip 

mkdir -p ../RNGDet/checkpoints
mv pretrain_spacenet/RNGDet/RNGDetNet_best.pt ../RNGDet/checkpoints

mkdir -p ../RNGDet_multi_ins/checkpoints
mv pretrain_spacenet/RNGDet_multi_ins/RNGDetNet_best.pt ../RNGDet_multi_ins/checkpoints