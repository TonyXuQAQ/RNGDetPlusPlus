declare -a arr=( $(jq -r '.test[]' ../data/data_split.json) )

# source directory
sudo python3 spacenet_convert.py
dir=RNGDet_multi_ins/test

echo $dir
mkdir -p ../${dir}/results/apls
# now loop through the above array
for i in "${arr[@]}"   
do
    pred_graph=${i}_crop.p
    gt_graph=${i}__gt_graph_dense_spacenet.p
    echo "========================$i======================"
    python ./apls/convert.py "../${dir}/graph/${pred_graph}" prop.json
    python ./apls/convert.py "../data/RGB_1.0_meter/${gt_graph}" gt.json
    sudo go run ./apls/main.go gt.json prop.json ../$dir/results/apls/$i.txt  spacenet
    # break
done
sudo python3 apls.py --dir $dir