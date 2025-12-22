CUDA_VISIBLE_DEVICES=1 nohup python -u -m codes.run.train \
  --config config/wikidiverse_stage2.yaml \
  > logs/wikidiverse_stage2.txt 2>&1 &