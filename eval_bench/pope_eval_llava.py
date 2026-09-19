import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
# print(sys.path)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
from PIL import Image
from torchvision.transforms import v2

from pope_loader import POPEDataSet
from eval_common import pope_parse, binary_metrics, done_keys, append_jsonl, read_jsonl, write_json, YESNO_MAX_NEW_TOKENS

# import kornia
from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default="/mnt/server8_hard1/donguk/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--data_path", type=str, default="/mnt/server18_hard0/jhjang/LVLM/crg/data/coco/val2014")
    parser.add_argument("--pope_path", type=str, default="/mnt/server8_hard1/donguk/rips2024/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/pope")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--enhance_layer_index", type=int, default=0)
    
    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.6)

    
    parser.add_argument("--max_new_tokens", type=int, default=YESNO_MAX_NEW_TOKENS)
    parser.add_argument("--greedy", type=str2bool, default=True)
    parser.add_argument("--out_path", type=str, default="./results/pope")
    parser.add_argument("--type", type=str, default="random")
    parser.add_argument("--dataset_name", type=str, default="coco")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()
    
    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        if args.use_ritual:
            method_name = "RITUAL"
        elif args.use_vcd:
            method_name = "VCD"
        elif args.use_m3id:
            method_name = "M3ID"
        elif args.use_only:
            method_name = "ONLY"
        else:
            method_name = "Regular"
        decoding = "greedy" if args.greedy else "sample"
        experiment_dir = f"{args.out_path}/llava-1.5-7b/{method_name}/{args.dataset_name}_{args.type}"
        run_tag = f"{decoding}_a{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_b{args.ritual_beta}_g{args.js_gamma}_L{args.enhance_layer_index}_T{args.max_new_tokens}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    pred_file = f"{experiment_dir}/{run_tag}_predictions.jsonl"
    metric_file = f"{experiment_dir}/{run_tag}_metrics.json"
    finished = done_keys(pred_file, "question_id")
    logger.info(f"predictions -> {pred_file} ({len(finished)} already done)")

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing Model')

    #### for ritual
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=image_processor,
        model=args.model_base
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    # ==============================================
    #               Augmentations
    # ==============================================

    aug_dict = {
    'horizontal flip':v2.RandomHorizontalFlip(p=1),
    'vertical flip':v2.RandomVerticalFlip(p=1),
    'rotation':v2.RandomRotation(degrees=180),
    'color jitter':v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
    'gaussian blur':v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
    'crop':v2.RandomResizedCrop(size=336),
    }
    
    # For statistics
    pos_aug_counter = {k:0 for k in aug_dict}
    pos_aug_counter.update({None: 0})

    # ========================================
    #            Start Generation
    # ========================================
    logger.info("Start eval...")
    vision_tower = model.get_vision_tower()
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        qid = data["question_id"][0]
        qid = qid.item() if torch.is_tensor(qid) else qid
        if qid in finished:
            continue
        image = data["image"][0]
        qs = data["query"][0]
        label = int(data["label"][0])
        image_path = data["image_path"]

        image_pos = None
        image_neg = None

        if args.use_ritual:
            raw_image = Image.open(image_path[0])
            pos_aug = random.choice(list(aug_dict.keys()))
            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0]
                image_pos = torch.tensor(image_pos)
            pos_aug_counter[pos_aug] += 1
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)

        # ==============================================
        #              Text prompt setting
        # ==============================================
        conv_out = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        # benchmark question verbatim (POPE protocol, also ONLY's own setting)
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()

        input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

        # ONLY's attention intervention hard-codes the image span as [35, 35+576) and keeps BOS (index 0)
        # in the sequence; make sure this prompt matches that layout.
        if args.use_only:
            img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()
            assert img_pos.numel() == 1 and img_pos.item() == 35 and vision_tower.num_patches == 576, \
                f"image span mismatch: start={img_pos.tolist()}, patches={vision_tower.num_patches}"
            assert input_ids[0, 0].item() == tokenizer.bos_token_id

        with torch.inference_mode():
            output_ids, _ = model.generate(
                input_ids,
                images=image.unsqueeze(0).half().cuda(),
                images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                do_sample=True,  # routes to the patched only_sample.sample; `greedy` selects argmax there
                greedy=args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                use_ritual=args.use_ritual,
                use_vcd=args.use_vcd,
                use_m3id=args.use_m3id,
                use_only=args.use_only,
                enhance_layer_index=args.enhance_layer_index,
                ritual_alpha_pos=args.ritual_alpha_pos,
                ritual_alpha_neg=args.ritual_alpha_neg,
                ritual_beta=args.ritual_beta,
                js_gamma=args.js_gamma,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        append_jsonl(pred_file, {
            "question_id": qid,
            "image": os.path.basename(image_path[0]),
            "question": qs,
            "label": "yes" if label == 1 else "no",
            "output": outputs,
            "pred": pope_parse(outputs),
        })

    records = read_jsonl(pred_file)
    assert len(records) == len(pope_dataset), f"{len(records)} predictions for {len(pope_dataset)} questions"
    metrics = binary_metrics([r["pred"] for r in records], [r["label"] for r in records])
    metrics["args"] = vars(args)
    write_json(metric_file, metrics)
    logger.info(
        f"[{args.type}] acc: {metrics['Accuracy']:.2f}, precision: {metrics['Precision']:.2f}, "
        f"recall: {metrics['Recall']:.2f}, f1: {metrics['F1']:.2f}, yes_ratio: {metrics['YesRatio']:.2f}"
    )


if __name__ == "__main__":
    main()
