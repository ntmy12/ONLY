"""BEAF evaluation for LLaVA-1.5 (+ONLY). Same decoding/prompt protocol as pope_eval_llava.py.

Answers are written in the official format [{"id", "answer"}] and scored by the unmodified
`beaf_metric.py` from https://github.com/postech-ami/BEAF.
"""
import os
import sys
import argparse
import subprocess

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from utils import dist_util
from utils.logger import create_logger
from eval_common import (load_beaf, beaf_image_index, beaf_official_answer, done_keys, append_jsonl,
                         read_jsonl, write_json, YESNO_MAX_NEW_TOKENS)

from only_utils.only_sample import evolve_only_sampling
evolve_only_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="BEAF evaluation on LLaVA-1.5.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--qna_path", type=str, required=True, help="beaf_qna.json")
    parser.add_argument("--image_root", type=str, required=True, help="folder containing all BEAF images")
    parser.add_argument("--beaf_metric", type=str, required=True, help="path to official beaf_metric.py")
    parser.add_argument("--out_path", type=str, default="./results/beaf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--enhance_layer_index", type=int, default=0)
    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.2)

    parser.add_argument("--greedy", type=str2bool, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=YESNO_MAX_NEW_TOKENS)
    return parser.parse_args()


class BEAFDataset(Dataset):
    def __init__(self, qna, image_index, image_processor, skip_ids):
        self.items = [q for q in qna if q["id"] not in skip_ids]
        self.image_index = image_index
        self.image_processor = image_processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        q = self.items[i]
        image = Image.open(self.image_index[q["image"]]).convert("RGB")
        pixel = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return {"id": q["id"], "image": pixel, "question": q["question"]}


def main():
    args = parse_args()
    dist_util.setup_dist(args)

    method_name = "ONLY" if args.use_only else "Regular"
    decoding = "greedy" if args.greedy else "sample"
    exp_dir = f"{args.out_path}/llava-1.5-7b/{method_name}"
    os.makedirs(exp_dir, exist_ok=True)
    run_tag = f"{decoding}_a{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_b{args.ritual_beta}_g{args.js_gamma}_L{args.enhance_layer_index}_T{args.max_new_tokens}"
    logger = create_logger(exp_dir)
    pred_file = f"{exp_dir}/{run_tag}_predictions.jsonl"
    answer_file = f"{exp_dir}/{run_tag}_answers.json"
    metric_file = f"{exp_dir}/{run_tag}_metrics.txt"

    qna = load_beaf(args.qna_path)
    image_index = beaf_image_index(args.image_root)
    missing = sorted({q["image"] for q in qna} - set(image_index))
    assert not missing, f"{len(missing)} BEAF images not found under {args.image_root}, e.g. {missing[:3]}"
    finished = done_keys(pred_file, "id")
    logger.info(f"{len(qna)} questions, {len(finished)} already done -> {pred_file}")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))
    vision_tower = model.get_vision_tower()

    loader = DataLoader(BEAFDataset(qna, image_index, image_processor, finished),
                        batch_size=1, shuffle=False, num_workers=args.num_workers)

    for data in tqdm(loader):
        qid = int(data["id"][0])
        qs = data["question"][0]
        conv = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"), version="v1", messages=[], offset=0,
            sep_style=SeparatorStyle.TWO, sep=" ", sep2="</s>",
        )
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + qs)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if args.use_only:
            img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()
            assert img_pos.numel() == 1 and img_pos.item() == 35 and vision_tower.num_patches == 576
            assert input_ids[0, 0].item() == tokenizer.bos_token_id

        with torch.inference_mode():
            output_ids, _ = model.generate(
                input_ids,
                images=data["image"].half().cuda(),
                images_pos=None,
                images_neg=None,
                do_sample=True,  # routes to only_sample.sample; `greedy` picks argmax there
                greedy=args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                use_ritual=False,
                use_vcd=False,
                use_m3id=False,
                use_only=args.use_only,
                enhance_layer_index=args.enhance_layer_index,
                ritual_alpha_pos=args.ritual_alpha_pos,
                ritual_alpha_neg=args.ritual_alpha_neg,
                ritual_beta=args.ritual_beta,
                js_gamma=args.js_gamma,
            )
        out = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if out.endswith(conv.sep2):
            out = out[:-len(conv.sep2)].strip()
        append_jsonl(pred_file, {"id": qid, "question": qs, "answer": out})

    preds = {r["id"]: r["answer"] for r in read_jsonl(pred_file)}
    assert len(preds) == len(qna), f"{len(preds)} answers for {len(qna)} questions"
    answers = [{"id": q["id"], "answer": preds[q["id"]]} for q in qna]  # same order as beaf_qna.json
    write_json(answer_file, answers)
    unparsable = [a["id"] for a in answers if beaf_official_answer(a["answer"]) is None]
    logger.info(f"answers without 'yes'/'no' (beaf_metric.py would reuse the previous answer): {len(unparsable)}")

    res = subprocess.run([sys.executable, args.beaf_metric, "--qna-path", args.qna_path, "--model-answers", answer_file],
                         capture_output=True, text=True)
    logger.info(res.stdout + res.stderr)
    with open(metric_file, "w") as f:
        f.write(res.stdout + res.stderr + f"\nunparsable_answers: {len(unparsable)}\n")


if __name__ == "__main__":
    main()
