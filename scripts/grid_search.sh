#!/bin/bash

# source ~/miniconda3/bin/activate sbert
source ~/anaconda3/bin/activate sbert
# Apache Android BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Windows Zookeeper 
datasets=( Apache Android BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Windows Zookeeper MultiSource MultiUnique )
# datasets=( MultiUnique )
models=( agglomerative )
# embeddings=( all-MiniLM-L12-v2 all-MiniLM-L6-v2 one-stage-ck10-MiniLM-L6-v2 two-stage-ck10-MiniLM-L6-v2 ) one-cdk5-m0.5-e5-b128-L6 one-cdk5-m0.05-e2-b128-L6 one-cdk3-m0.5-e5-b128-L6 one-cdk3-m0.05-e2-b128-L6 
embeddings=( MultiSource-full-crk0-m0.5-e5-b128-L6 MultiSource-full-crk1-m0.5-e5-b128-L6 MultiSource-full-cdk1-m0.5-e5-b128-L6 MultiSource-full-crk3-m0.5-e5-b128-L6 MultiSource-full-cdk3-m0.5-e5-b128-L6 MultiSource-full-crk5-m0.5-e5-b128-L6 MultiSource-full-cdk5-m0.5-e5-b128-L6 MultiSource-full-crk10-m0.5-e5-b128-L6 MultiSource-full-cdk10-m0.5-e5-b128-L6 ) # one-crk10-m0.5-e5-b128-L6 one-crk10-m0.05-e2-b128-L6 one-cdk10-m0.5-e5-b16-v1 
# embeddings=( all-mpnet-base-v2 all-MiniLM-L12-v2 all-MiniLM-L6-v2 one-stage-k10-MiniLM-L6-v2 two-stage-k10-MiniLM-L6-v2 one-stage-k10-MiniLM-L12-v2 two-stage-k10-MiniLM-L12-v2 one-stage-k5-MiniLM-L12-v2 two-stage-k5-MiniLM-L12-v2 one-stage-k3-MiniLM-L12-v2 two-stage-k3-MiniLM-L12-v2 )
# thresholds=( 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.9 0.95 1 )
thresholds=( 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 )
for embedding in "${embeddings[@]}"; do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            for threshold in "${thresholds[@]}"; do
                python -m src.lasec --dataset "$dataset" --model "$model" --embedding "$embedding" --threshold "$threshold" --output_dir grid-full
                # python -m src.lasec --dataset "$dataset" --model "$model" --embedding "$embedding" --threshold "$threshold" --output_dir effect-k --held_out
            done
        done
    done
done
for embedding in "${embeddings[@]}"; do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            for threshold in "${thresholds[@]}"; do
                # python -m src.lasec --dataset "$dataset" --model "$model" --embedding "$embedding" --threshold "$threshold" --output_dir effect-k
                python -m src.lasec --dataset "$dataset" --model "$model" --embedding "$embedding" --threshold "$threshold" --output_dir grid-ho --held_out
            done
        done
    done
done