num_gpu=4
max=3
for ((i=0; i < $num_gpu; i++)); do
    for ((j=0; j < $max; j++)); do
        CUDA_VISIBLE_DEVICES=$i $1 $2 $3 &!
    done
done
wait