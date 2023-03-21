# for dim in 16
# do
#     for seed in {0..4}
#     do 
#         python node2vec.py --embedding_dim $dim --seed $seed
#     done
# done


for seed in {0..4}
do 
    python node2vec.py --embedding_dim 16 --seed $seed
done