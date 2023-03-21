# for dim in 8 16 32 64 128 256
# do
#     for seed in {0..4}
#     do 
#         python mlp.py --use_node_embedding --dim $dim --seed_id $seed
#     done
# done

# for dim in 16
# do
#     for seed in {0..4}
#     do 
#         python mlp.py --use_node_embedding --dim $dim --seed_id $seed
#     done
# done

for dim in 16
do
    for seed in {0..4}
    do 
        python mlp_no_concat.py --use_node_embedding --dim $dim --seed_id $seed
    done
done
