#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py     --config configs \
                                                --datasets V\
                                                --backbone ViT-B/16 \
                                                --topk 1
                                                