#!/bin/bash
# generate PromptCase embeddings for COLIEE 2017 train and test
python PromptCase/promptcase_embedding/PromptCase_embedding_generation.py --data 2017 --dataset test
python PromptCase/promptcase_embedding/PromptCase_embedding_generation.py --data 2017 --dataset train
