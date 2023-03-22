for dim in 8 16 32 64 128 256
do
    for seed in {0..9}
    do 
        python generate_embeddings.py --embedding_dim $dim --seed $seed
    done
done
