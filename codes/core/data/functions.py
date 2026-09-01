import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import ujson


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument(
        "--candidates_out_dir",
        type=str,
        default="./data/processed",
        help="Output directory for generated top-k candidates (jsonl)."
    )
    parser.add_argument(
        "--candidates_topk",
        type=int,
        default=10,
        help="Top-k candidates to generate for each sample."
    )

    cli_args = parser.parse_args()
    cfg = OmegaConf.load(cli_args.config)

    override = OmegaConf.create({
        "candidates_out_dir": cli_args.candidates_out_dir,
        "candidates_topk": cli_args.candidates_topk,
    })
    cfg = OmegaConf.merge(cfg, override)

    return cfg

def load_jsonl_file(filepath, desc='', key=None):
    if key is None:
        data = []
    else:
        data = dict()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=desc):
            item = ujson.loads(line)
            if key:
                item_key = item.get(key)
                data[item_key] = item
            else:
                data.append(item)
    return data