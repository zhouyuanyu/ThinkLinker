CUDA_VISIBLE_DEVICES=1 nohup python -u -m codes.run.train \
  --config config/wikidiverse_stage1.yaml \
  > logs/wikidiverse_stage1.txt 2>&1 &