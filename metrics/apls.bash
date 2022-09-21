declare -a arr=(8 9 19 28 29 39 48 49 59 68 69 79 88 89 99 108 109 119 128 129 139 148 149 159 168 169 179)

# output directory
dir=RNGDet/test

echo $dir
# now loop through the above array
for i in "${arr[@]}"   
do
    if test -f "../${dir}/graph/${i}.p"; then
        echo "========================$i======================"
        python ./apls/convert.py "../data/20cities/region_${i}_graph_gt.pickle" gt.json
        python ./apls/convert.py "../${dir}/graph/${i}.p" prop.json
        sudo go run ./apls/main.go gt.json prop.json ../$dir/results/apls/$i.txt 
    fi
done
sudo python3 apls.py --dir $dir