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

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
from PIL import Image
from torchvision.transforms import v2

from chair_loader import CHAIRDataset
from eval_common import select_chair_images, done_keys, append_jsonl, CHAIR_PROMPT, CHAIR_MAX_NEW_TOKENS, CHAIR_NUM_IMAGES

# import kornia
from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings(action='ignore')

import time

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
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, help="model")
    parser.add_argument("--model_base", type=str, default="llava")

    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--data_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/val2014/", help="data path")
    parser.add_argument("--anno_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/annotations/instances_val2014.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/chair")
    parser.add_argument("--out_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/chair_results/llava", help="output path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--method_name", type=str, default='none')
    parser.add_argument("--enhance_layer_index", type=int, default=0)

    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.1)

    parser.add_argument("--num_eval_samples", type=int, default=CHAIR_NUM_IMAGES)
    parser.add_argument("--max_new_tokens", type=int, default=CHAIR_MAX_NEW_TOKENS)
    parser.add_argument("--prompt", type=str, default=CHAIR_PROMPT)
    parser.add_argument("--chair_seed", type=int, required=True, help="seed that selects the CHAIR images")
    parser.add_argument("--image_list", type=str, default=None,
                        help="json with the selected images; created on first use and reused afterwards")
    parser.add_argument("--greedy", type=str2bool, default=True)

    args = parser.parse_known_args()[0]
    return args


def main():
    args = parse_args()
    # print(args)
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing Model')

    #### for ritual
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    image_list = args.image_list or os.path.join(os.path.dirname(os.path.abspath(args.out_path)), f"chair_images_seed{args.chair_seed}.json")
    img_files = select_chair_images(args.data_path, args.chair_seed, args.num_eval_samples, image_list)
    chair_dataset = CHAIRDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        trans=image_processor,
        model=args.model_base,
        img_files=img_files,
    )
    chair_loader = DataLoader(
        chair_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    os.makedirs(
        args.out_path, exist_ok=True
    )
    decoding = "greedy" if args.greedy else "sample"
    method_tag = args.method_name if args.use_only or args.use_vcd or args.use_ritual or args.use_m3id else "regular"
    cap_file = os.path.join(args.out_path, f"{method_tag}_{decoding}_a{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_b{args.ritual_beta}_g{args.js_gamma}_L{args.enhance_layer_index}_T{args.max_new_tokens}_seed{args.chair_seed}.jsonl")
    finished = done_keys(cap_file, "image_id")
    logger.info(f"captions -> {cap_file} ({len(finished)} already done)")
    vision_tower = model.get_vision_tower()



    # ========================================
    #            Start Generation
    # ========================================
    logger.info("Start eval...")
    for batch_id, data in tqdm(enumerate(chair_loader), total=args.num_eval_samples):

        # early stop for debuggging purpose
        # if batch_id == 20:
        #     break

        if batch_id == args.num_eval_samples:
            break
            
        img_id = data["image_id"]
        if img_id.item() in finished:
            continue
        image_path = data["image_path"]
        image = data["image"]

        qs = args.prompt

        image_pos = None
        image_neg = None
        
        if args.use_ritual:
            # ==============================================
            #              Image Transforms
            # ==============================================
            raw_image = Image.open(image_path[0])
            pos_aug = random.choice(list(aug_dict.keys()))

            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0] 
                image_pos = torch.tensor(image_pos)
                
            pos_aug_counter[pos_aug] += 1
            logger.info(f"RITUAL Transformation: {pos_aug}")
        
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)
        

        # ==============================================
        #              Text prompt setting
        # ==============================================
        t1 = time.time()
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
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()

        # ==============================================
        #             Image tensor setting
        # ==============================================
        input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if args.use_only:
            # ONLY hard-codes the image span [35, 35+576) with BOS kept at index 0
            img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()
            assert img_pos.numel() == 1 and img_pos.item() == 35 and vision_tower.num_patches == 576
            assert input_ids[0, 0].item() == tokenizer.bos_token_id

        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

        # ==============================================
        #                ritual method
        # ==============================================
        with torch.inference_mode():
            with torch.no_grad():
                output_ids, _ = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                    images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                    do_sample=True,  # routes to only_sample.sample; `greedy` picks argmax there
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
                    ritual_alpha_pos=args.ritual_alpha_pos,
                    ritual_alpha_neg=args.ritual_alpha_neg,
                    ritual_beta=args.ritual_beta,
                    js_gamma=args.js_gamma,
                    enhance_layer_index=args.enhance_layer_index,
                )
        t2 = time.time()
        print(f"Time: {t2-t1}")
                
        input_token_len = input_ids.shape[1]
        gen_ids = output_ids[0, input_token_len:].tolist()
        # generated length in tokens, excluding the terminating EOS
        num_tokens = len(gen_ids) - (1 if gen_ids and gen_ids[-1] == tokenizer.eos_token_id else 0)
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        append_jsonl(cap_file, {
            "image_id": img_id.item(),
            "image": os.path.basename(image_path[0]),
            "caption": outputs,
            "num_tokens": num_tokens,
            "time": t2 - t1,
        })

    # logger.info(vars(args))

    # if args.use_ritual:
    #     logger.info(f"RITUAL Transformation: {pos_aug_counter}")

if __name__ == "__main__":
    main()