#!/bin/sh

python train_student.py \
  --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
  --distill gkdfd \
  --model_s resnet8x4 \
  --batch_size 64 \
  --gkdfd_k1 7 \
  --gkdfd_k2 56 \
  --trial 1
