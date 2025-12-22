import os
import sys
import json
import time
import datetime
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from codes.core.data.functions import setup_parser
from codes.core.model.lit_thinklinker import ThinkLinkerLitModule
from codes.core.data.dataset import DataModule
from torch.utils.data import DataLoader

# --config
# first_stage_ckpt
# --candidates_out_dir
# --candidates_topk

@torch.no_grad()
def _score_and_topk_from_batch(lightning_model, batch: Dict[str, Any], k: int, device: torch.device ):
    mention_idx = batch["mention_idx"]
    mention_input_dict = batch["mention_input_dict"]
    entity_input_dict = batch["entity_input_dict"]
    candidate_qids = batch.get("candidate_qids", None)
    mention_id = batch.get("mention_id", None)

    mention_idx = mention_idx.to(device)
    mention_input_dict = mention_input_dict.to(device)
    entity_input_dict = entity_input_dict.to(device)

    mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
        lightning_model.encoder(**mention_input_dict)

    entity_text_cls, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
        lightning_model.encoder(**entity_input_dict)

    batch_size = mention_text_embeds.shape[0]
    total_entity = entity_text_cls.shape[0]

    if total_entity % batch_size != 0:
        raise ValueError(
            f"total_entity ({total_entity}) is not divisible by batch_size ({batch_size}). "
            f"Cannot reshape entity tensors into [B, num_cands, ...]."
        )
    num_cands = total_entity // batch_size

    entity_text_cls = entity_text_cls.reshape(batch_size, num_cands, -1)
    entity_image_embeds = entity_image_embeds.reshape(batch_size, num_cands, -1)

    length, dim = entity_text_seq_tokens.shape[-2:]
    entity_text_seq_tokens = entity_text_seq_tokens.reshape(batch_size, num_cands, length, dim)

    length, dim = entity_image_patch_tokens.shape[-2:]
    entity_image_patch_tokens = entity_image_patch_tokens.reshape(batch_size, num_cands, length, dim)

    score, _ = lightning_model.matcher(
        entity_text_cls,
        entity_text_seq_tokens,
        mention_text_embeds,
        mention_text_seq_tokens,
        entity_image_embeds,
        entity_image_patch_tokens,
        mention_image_embeds,
        mention_image_patch_tokens
    )

    if score.dim() != 2:
        raise ValueError(f"Expected score shape [B, num_cands], got {tuple(score.shape)}")

    # top-k
    k = min(int(k), score.shape[1])
    topv, topi = torch.topk(score, k=k, dim=-1)

    topi_list = topi.detach().cpu().tolist()
    top_scores = topv.detach().cpu().tolist()
    mention_idxs = mention_idx.detach().cpu().tolist()

    if mention_id is None:
        mention_ids = [str(x) for x in mention_idxs]
    else:
        if torch.is_tensor(mention_id):
            mention_ids = [str(x) for x in mention_id.detach().cpu().tolist()]
        else:
            mention_ids = [str(x) for x in mention_id]
            if len(mention_ids) != batch_size:
                raise ValueError(f"mention_id length mismatch: {len(mention_ids)} vs batch_size={batch_size}")

    cand_grid = None
    if candidate_qids is not None:
        if isinstance(candidate_qids, list) and len(candidate_qids) > 0 and isinstance(candidate_qids[0], list):
            cand_grid = candidate_qids
        else:
            flat = list(candidate_qids)
            if len(flat) != batch_size * num_cands:
                raise ValueError(
                    f"candidate_qids length mismatch: got {len(flat)}, expected {batch_size * num_cands} "
                    f"(B={batch_size}, num_cands={num_cands})."
                )
            cand_grid = [flat[i * num_cands:(i + 1) * num_cands] for i in range(batch_size)]

    lines = []
    for i in range(batch_size):
        if cand_grid is None:
            top_qids = [int(x) for x in topi_list[i]]
        else:
            top_qids = [cand_grid[i][local_idx] for local_idx in topi_list[i]]

        lines.append({
            "idx": int(mention_idxs[i]),
            "mention_id": str(mention_ids[i]),
            "topk_qids": top_qids,
            "topk_scores": top_scores[i]
        })

    rank = torch.argsort(torch.argsort(score, dim=-1, descending=True), dim=-1, descending=False) + 1
    tgt_rank = rank[torch.arange(score.shape[0]), 0]  # 你原来取第0列为 tgt
    tgt_rank = tgt_rank.detach().cpu()

    return lines, tgt_rank


@torch.no_grad()
def generate_candidates_for_split(lightning_model, dataloader, split_name: str, out_path: str, device: torch.device, k: int = 3):
    lightning_model.eval()
    lightning_model.to(device)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    written = 0
    for step, batch in enumerate(dataloader):
        lines, _ = _score_and_topk_from_batch(lightning_model, batch, k=k, device=device)

        with open(out_path, "a", encoding="utf-8") as fw:
            for obj in lines:
                fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

        written += len(lines)
        if (step + 1) % 50 == 0:
            print(f"[{split_name}] step={step+1} written={written}")
            sys.stdout.flush()

    print(f"[OK] {split_name}: saved {written} lines -> {out_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    start_ts = time.time()
    print(f"========== START: {datetime.datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S')} ==========")
    sys.stdout.flush()

    args = setup_parser()
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(1)

    data_module = DataModule(args)
    lightning_model = ThinkLinkerLitModule(args)

    data_module.setup(stage="fit")
    train_loader_for_gen = DataLoader(
        data_module.train_data,
        batch_size=args.data.eval_batch_size,
        num_workers=args.data.num_workers,
        shuffle=False,
        collate_fn=lambda x: data_module.train_collator(x, is_eval=True),
    )
    val_loader = data_module.val_dataloader()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    if not getattr(args, "first_stage_ckpt", None):
        raise ValueError("Please provide --first_stage_ckpt to generate candidates.")
    if not os.path.isfile(args.first_stage_ckpt):
        raise FileNotFoundError(f"--first_stage_ckpt not found: {args.first_stage_ckpt}")

    print(f"[INFO] Loading checkpoint weights from: {args.first_stage_ckpt}")
    sys.stdout.flush()
    ckpt = torch.load(args.first_stage_ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    lightning_model.load_state_dict(state_dict, strict=False)


    out_dir = getattr(args, "candidates_out_dir", "./data/processed")
    k = int(getattr(args, "candidates_topk", 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 候选实体生成
    generate_candidates_for_split(
        lightning_model,
        train_loader_for_gen,
        split_name="train",
        out_path=os.path.join(out_dir, f"train_top{k}.jsonl"),
        k=k,
        device=device,
    )
    generate_candidates_for_split(
        lightning_model,
        val_loader,
        split_name="val",
        out_path=os.path.join(out_dir, f"val_top{k}.jsonl"),
        k=k,
        device=device,
    )
    generate_candidates_for_split(
        lightning_model,
        test_loader,
        split_name="test",
        out_path=os.path.join(out_dir, f"test_top{k}.jsonl"),
        k=k,
        device=device,
    )


    end_ts = time.time()
    dur = int(end_ts - start_ts)
    h, m, s = dur // 3600, (dur % 3600) // 60, dur % 60
    print(
        f"========== END: {datetime.datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Duration: {h:02d}:{m:02d}:{s:02d} ({dur} seconds) =========="
    )
    sys.stdout.flush()
