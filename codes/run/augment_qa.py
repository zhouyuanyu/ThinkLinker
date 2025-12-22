import os
import json
import argparse
from pathlib import Path
import gc

from dotenv import load_dotenv
from tqdm import tqdm
import logging

from codes.core.augmentation.answerer import Answerer
from codes.core.augmentation.questioner import Questioner


logger = logging.getLogger("generate_qa_per_candidate")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                logger.warning(f"跳过无法解析的行（{path}）：{e}\n-> {line[:200]}")
    return data


def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                logger.warning(f"跳过无法解析的行（{path}）：{e}\n-> {line[:200]}")
                continue


def build_kb_map(kb_list):
    kb_map = {}
    for e in tqdm(kb_list, desc="Processing kb_list", unit="item"):
        qid = str(e.get("qid"))
        if not qid:
            continue
        entity_name = e.get("name")
        desc = e.get("desc") or ""
        kb_map[qid] = {
            "qid": qid,
            "entity_name": entity_name,
            "desc": desc
        }
    return kb_map


def build_candidates_map(candidates_list):
    candidates_map = {}
    for e in tqdm(candidates_list, desc="Processing candidates_list", unit="item"):
        mention_id = str(e.get("mention_id"))
        if not mention_id:
            continue

        candidates_map[mention_id] = e
    return candidates_map


def normalize_candidates(candidates_list, max_per_mention=3):
    temp_map = {}
    for entry in candidates_list:
        mid = entry.get("mention_id")
        if mid is None:
            logger.debug(f"跳过无 mention id 的候选条目: {entry}")
            continue
        mid = str(mid)

        tq = entry.get("topk_qids")
        ts = entry.get("topk_scores")

        qids = [str(x) for x in tq]
        scores = []
        for i in range(len(qids)):
            if i < len(ts):
                try:
                    scores.append(float(ts[i]))
                except Exception:
                    scores.append(None)
            else:
                scores.append(None)
        if qids:
            temp_map.setdefault(mid, []).append({"qids": qids, "scores": scores})

    normalized = {}
    for mid, blocks in temp_map.items():
        all_qids = []
        all_scores = []
        for b in blocks:
            all_qids.extend(b["qids"])
            all_scores.extend(b["scores"])
        all_qids = all_qids[:max_per_mention]
        all_scores = (all_scores[:max_per_mention] + [None] * max(0, max_per_mention - len(all_scores)))
        normalized[mid] = [{"qid": all_qids[i], "score": all_scores[i]} for i in range(len(all_qids))]
    return normalized

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mentions", required=True,
                        help="mentions.jsonl")
    parser.add_argument("--candidates", required=True,
                        help="train_topk3.jsonl")
    parser.add_argument("--kb", required=True, help="Knowledge base file")
    parser.add_argument("--output", required=True, help="Output path for the mention file with enhanced text")
    parser.add_argument("--topk", type=int, default=8, help="Select the top-k candidates for each mention")
    parser.add_argument("--debug", action="store_true", help="Raise exceptions for debugging")
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to the file that stores the API key",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_file_path = args.env_file
    load_dotenv(env_file_path)

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("请在环境变量中设置 API_KEY（或 OPENAI_API_KEY），例如：export API_KEY=你的key")

    logger.info("加载输入文件...")
    mentions_list = load_jsonl(args.mentions)
    candidates_list = load_jsonl(args.candidates)

    logger.info(f"读取到 mentions: {len(mentions_list)} 条，candidates: {len(candidates_list)} 条（kb 文件将按需读取）")

    normalized_candidates = normalize_candidates(candidates_list, max_per_mention=args.topk)
    candidates_map = normalized_candidates

    needed_qids = set()
    for mid, items in candidates_map.items():
        for it in items:
            needed_qids.add(str(it["qid"]))

    logger.info(f"需要从 kb 中加载的 qid 数量: {len(needed_qids)}")

    kb_map = {}
    for e in tqdm(stream_jsonl(args.kb), desc="Building filtered kb_map"):
        qid = e.get("qid")
        if qid is None:
            continue
        qid = str(qid)
        if qid not in needed_qids:
            continue
        entity_name = e.get("name")
        desc = e.get("desc")
        kb_map[qid] = {"qid": qid, "entity_name": entity_name, "desc": desc}

    logger.info(f"kb_map 构建完成，实际加载到内存的实体数: {len(kb_map)}")

    del candidates_list
    gc.collect()

    logger.info("初始化 Answerer 和 Questioner ...")
    vqa = Answerer(os.environ["API_KEY"])
    questioner = Questioner(api_key=os.environ["API_KEY"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)
    fout = open(out_path, "a", encoding="utf-8")

    for mention_row in tqdm(mentions_list, desc="Processing mentions"):
        mid = str(mention_row.get("mention_id"))

        out_record = {
            ** mention_row,
            "generated": []
        }

        cand_list = candidates_map.get(mid, [])
        if not cand_list:
            logger.info(f"mention {mid} 没有候选（candidates 文件中找不到），直接写出空 generated")
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fout.flush()
            continue

        questioner.reset_conversation(mention_id=mid,
                                      mention=mention_row.get("mentions"))

        qid = cand_list[0]["qid"]
        anchor_candidate = kb_map.get(str(qid))

        logger.info(f"Generating question for round {1}")

        qres = questioner.generate_question(anchor_candicate=anchor_candidate)
        if isinstance(qres, dict):
            question_text = qres.get("question")
        else:
            question_text = qres

        for i in range(1, args.topk + 1):
            try:
                answer_text = vqa.process_question(mention_row, question_text)
            except Exception as e:
                logger.warning(f"调用 Answerer 失败 (mention {mid}, qid {qid})：{e}")
                answer_text = ""

            single_gen = {
                "qid": qid,
                "entity_name": anchor_candidate['entity_name'],
                "question": question_text,
                "answer": answer_text
            }

            if i < len(cand_list):
                qid = cand_list[i]["qid"]
            else:
                logger.info(f"mention {mid} 的候选不足 topk（索引 {i} 不存在），停止本 mention 的循环")
                out_record["generated"].append(single_gen)
                break

            anchor_candidate = kb_map.get(str(qid))

            questioner.record_answer(reranked_top1_entity=anchor_candidate, answer=question_text)

            is_last_round = (i == args.topk)

            if not is_last_round:
                logger.info(f"Generating question for round {i + 1}")
                qres = questioner.generate_question(anchor_candicate=anchor_candidate, temperature=0.7)
                if isinstance(qres, dict):
                    question_text = qres.get("question")
                else:
                    question_text = qres
                if question_text is None:
                    logger.info(f"Failed to generate question for round {i + 2}")
                    failed_generation = True
                    break
            else:
                logger.info(f"Final round {i + 1} completed, no more questions will be generated")

            out_record["generated"].append(single_gen)

        fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    logger.info(f"全部完成，输出文件：{args.output}")


if __name__ == "__main__":
    main()
