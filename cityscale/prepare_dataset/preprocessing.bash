# Download sat2graph dataset
# # This link is invalid now: wget https://mapster.csail.mit.edu/sat2graph/data.zip
# if you cannot download this by script, download it manually at https://drive.google.com/file/d/1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H/view?usp=share_link
# gdown https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H
# !!!!!!!!!1 Google Drive links have failed, download it manually from https://cloud.tsinghua.edu.cn/d/d32cb7d4b19046ed9a42/ !!!!!!!!!!!!!
unzip data.zip
rm -rf data.zip 
mkdir -p ../data
mv ./data/* ../data

# Generate label
echo "Generating labels ..."
python create_label.py
python data_split.py
echo "Finsh generating labels!"

# Get pretrained checkpoints
# if you cannot download this by script, download it manually at https://drive.google.com/file/d/1AwlFt06GRJaHKk1rboATKjv2qK5LnEvn/view?usp=share_link
# gdown https://drive.google.com/uc?id=1AwlFt06GRJaHKk1rboATKjv2qK5LnEvn
# !!!!!!!!!1 Google Drive links have failed, download it manually from https://cloud.tsinghua.edu.cn/d/d32cb7d4b19046ed9a42/ !!!!!!!!!!!!!
unzip pretrain_cityscale.zip
rm -rf pretrain_cityscale.zip 

mkdir -p ../RNGDet/checkpoints
mv pretrain_cityscale/RNGDet/RNGDet_best.pt ../RNGDet/checkpoints

mkdir -p ../RNGDet_multi_ins/checkpoints
mv pretrain_cityscale/RNGDet_multi_ins/RNGDetPP_best.pt ../RNGDet_multi_ins/checkpoints
