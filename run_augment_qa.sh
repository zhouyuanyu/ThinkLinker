CUDA_VISIBLE_DEVICES=1 nohup python -u -m codes.run.augment_qa \
  --mentions  data/WikiDiverse/WIKIDIVERSE_train.jsonl \
  --candidates data/WikiDiverse/processed/train_top3.jsonl \
  --kb data/WikiDiverse/entities.jsonl \
  --output data/WikiDiverse/processed/train_enhanced.jsonl \
  --topk 3 \
  --env_file .env \
  > logs/wikidiverse_gen_txt.txt 2>&1 &