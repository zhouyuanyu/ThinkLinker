CUDA_VISIBLE_DEVICES=1 nohup python -u -m codes.run.gen_candidates \
  --config config/wikidiverse_stage1.yaml \
  --candidates_out_dir data/wikidiverse/processed \
  --candidates_topk 3 \
  > logs/wikidiverse_gen_candidates 2>&1 &