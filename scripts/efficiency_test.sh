#!/bin/bash
# BGL dataset
sample_sizes=( 5000 10000 50000 100000 500000 800000 1000000 )
embeddings=( all-MiniLM-L6-v2 all-MiniLM-L12-v2 all-distilroberta-v1 all-mpnet-base-v2 )


for embedding in "${embeddings[@]}"; do
    for size in "${sample_sizes[@]}"; do
        timeout 10001 python -m src.lasec --dataset_type full --dataset BGL --sample_size "$size" --model agglomerative --embedding "$embedding" --threshold 0.09 --output_dir efficiency-cpu --device cpu
    done
done

