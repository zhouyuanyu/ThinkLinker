# ThinkLinker: From Low-Rank Interaction to Knowledge-Aware Verification for Multimodal Entity Linking



## Dependencies

We recommend using Conda to manage virtual environments, and we use Python version **3.9.25**.

```python
conda create -n thinklinker python==3.9.25
conda activate thinklinker
pip install -r requirements.txt
```

Please install the specified versions of Python libraries according to the `requirements.txt` file. 



## Dataset

1. Download the datasets from [FissFuse paper]([pengfei-luo/FissFuse: [ACM MM 2024\] Bridging Gaps in Content and Knowledge for Multimodal Entity Linking](https://github.com/pengfei-luo/FissFuse)).
2. Create a data root directory at `./data/` and place the downloaded datasets under this directory.



## Running the code

### Step1. Stage-1 training

We provide separate configuration files for different datasets. Before running, please update all file/folder paths in the corresponding stage-1 config, e.g., `./config/wikidiverse_stage1.yaml`.

Run stage-1 training:

```bash
bash run_train.sh
```

Checkpoints will be saved under `./runs/` by default.



### Step2. Candidate generation

Set `first_stage_ckpt` in the stage-1 config to the checkpoint produced in Step 1. Then specify:

```
--candidates_out_dir data/wikidiverse/processed 
--candidates_topk 3 
```

Run:

```bash
bash run_gen_candidates.sh
```

This will produce `train_top3.jsonl`, `val_top3.jsonl`, and `test_top3.jsonl` under the output directory.



### Step3. LLM-based KS-SS Dialogic Verification

Configure your API key and LLM settings in `.env`, then provide the following arguments (example for training split):

```bash
--mentions  data/WikiDiverse/WIKIDIVERSE_train.jsonl 
--candidates data/WikiDiverse/processed/train_top3.jsonl 
--kb data/WikiDiverse/entities.jsonl 
--output data/WikiDiverse/processed/train_enhanced.jsonl 
--topk 3 
--env_file .env
```

Run:

```bash
bash run_augment_qa.sh
```

Repeat for train/val/test to obtain three `*_enhanced.jsonl` files.



### Step4. Stage-2 finetuning

Update paths in the corresponding stage-2 config (e.g., `./config/wikidiverse_stage2.yaml`). Then run:

```bash
bash run_finetune.sh
```

This performs stage-2 finetuning and produces the final results.

