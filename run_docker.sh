#!/bin/bash

HOME='/home/zeng/project'

#check for permission
script="train.sh"
if [ ! -x "$script" ]; then
    echo "Adding execute permission to $script"
    chmod +x "$script"
fi

docker run \
    --rm \
    -v $HOME:$HOME \
    -w $HOME \
    --gpus '"device='$CUDA_VISIBLE_DEVICES'"' \
    workenv:latest \
    $HOME/train.sh \
    "$@"