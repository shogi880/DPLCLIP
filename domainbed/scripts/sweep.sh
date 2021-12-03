num_gpu=$1
for ((i=0; i < $num_gpu; i++)); do
    CUDA_VISIBLE_DEVICES=$i $2 $3 $4 &!
    sleep 5
done
wait
