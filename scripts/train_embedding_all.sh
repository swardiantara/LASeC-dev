#!/bin/bash

# source ~/miniconda3/bin/activate sbert
# source ~/anaconda3/bin/activate sbert
initial_models=( all-MiniLM-L6-v2 ) # chosen embedding model
samplings=( random distance ) # on both sampling strategies
num_samples=( 0 1 3 5 10 ) # effect of k values
# margins=( 0.1 0.15 0.2 )

# python multi_stage_embedding.py --initial_model_path all-MiniLM-L6-v2 --sampling_strategy distance --stage one --k 10 --m2 "$margin" --push_embedding
# for margin in "${margins[@]}"; do
# done
# for margin in "${margins[@]}"; do
# done

#     python multi_stage_embedding.py --initial_model_path all-MiniLM-L6-v2 --sampling_strategy distance --stage one --k "$k" --push_embedding
# python multi_stage_embedding.py --initial_model_path all-MiniLM-L6-v2 --sampling_strategy distance --stage one --k "$k" --push_embedding
# python -m src.train_embedding --initial_model_path all-MiniLM-L6-v2 --sampling_strategy random --stage one --k 10 --template_portion partial --push_embedding
for model in "${initial_models[@]}"; do
    for k in "${num_samples[@]}"; do
        for sampling in "${samplings[@]}"; do
            python -m src.train_embedding --initial_model_path "$model" --sampling_strategy "$sampling" --k "$k" --push_embedding
        done
    done
done

initial_models=( all-MiniLM-L6-v2 all-MiniLM-L12-v2 all-mpnet-base-v2 all-distilroberta-v1 ) # effect of different initial models
samplings=( random )
num_samples=( 10 )

for model in "${initial_models[@]}"; do
    for k in "${num_samples[@]}"; do
        for sampling in "${samplings[@]}"; do
            python -m src.train_embedding --initial_model_path "$model" --sampling_strategy "$sampling" --k "$k" --push_embedding
        done
    done
done

# train with 80% of data
model="all-MiniLM-L6-v2"
sampling="random"
k=10
python -m src.train_embedding --initial_model_path "$model" --sampling_strategy "$sampling" --k "$k" --push_embedding --template_portion partial
