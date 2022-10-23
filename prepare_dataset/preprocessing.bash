# Download sat2graph dataset
# # This link is invalid now: wget https://mapster.csail.mit.edu/sat2graph/data.zip
gdown https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H
unzip data.zip
rm -rf data.zip 
mv ./data ../

# Generate label
echo "Generating labels ..."
python create_label.py
python data_split.py
echo "Finsh generating labels!"

# Get pretrained checkpoints
gdown https://drive.google.com/uc?id=1L2NWUJlFkS1vyqhbzY33fjjAHRi-5189
mkdir -p ../RNGDet/checkpoints
mv RNGDet_best.pt ../RNGDet/checkpoints

gdown https://drive.google.com/uc?id=1yUzCi_yVNO9YFKfJ9ITTt5KLxisBD9ol
mkdir -p ../RNGDet_multi_ins/checkpoints
mv RNGDetPP_best.pt ../RNGDet_multi_ins/checkpoints
