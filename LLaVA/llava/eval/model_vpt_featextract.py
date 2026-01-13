import argparse
import torch
import os
import PIL
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, perspective_data=args.perspective_data)

    # for name, module in model.named_modules():
    #    print(name, "→", module)

    ###### NEW ####### forward hooks
    vision_tower = model.model.vision_tower.vision_tower.vision_model.encoder.layers[-1].self_attn.k_proj
    multimodal_projector = model.model.mm_projector[2]
    language_model = model.model.layers[-1].self_attn.q_proj

    # create dict to store features
    features = {
        "vision_tower": [],
        "multimodal_projector": [],
        "language_model": []
    }

    def get_features(name):
        def hook(model, input, output):
            output = output.detach().cpu()
            print("Output shape:", output.shape)

            features[name].append(output.numpy().squeeze())
        return hook

    vision_tower.register_forward_hook(get_features("vision_tower"))
    multimodal_projector.register_forward_hook(get_features("multimodal_projector"))
    language_model.register_forward_hook(get_features("language_model"))

    ################

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        print(model.config.mm_use_im_start_end)

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        print("qs:", qs)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # prompt = qs
        print("Prompt:", prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        ###### NEW #######
        image = image.resize((336,336), resample = PIL.Image.NEAREST )
        ###### NEW #######
        image_tensor = process_images([image], image_processor, model.config)[0]

        print("input_ids shape:", input_ids.shape)
        print("image_tensor shape:", image_tensor.shape)
        print("image_sizes:", image.size)

        assert not torch.isnan(image_tensor).any(), "NaNs in image tensor!"
        assert not torch.isinf(image_tensor).any(), "Infs in image tensor!"

        prompt_len = input_ids.shape[1] # length of the prompt

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        print("output_ids:", output_ids)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"outputs: {outputs}")

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

    # Save features
    # features are in dict with keys: vision_tower, multimodal_projector, language_model
    # values are lists of numpy arrays, let's make an np array of shape (n_layers, n_samples, n_features)
    vision_tower_features = np.array(features["vision_tower"])
    multimodal_projector_features = np.array(features["multimodal_projector"])
    
    # print(len(features["language_model"]))
    for i, arr in enumerate(features["language_model"]):
         print(f"Index {i}: shape {arr.shape}")
    cleaned_lang_features = [
        arr for arr in features["language_model"] if arr.shape[0] == 645
        ]
    language_model_features = np.array(cleaned_lang_features)

    os.makedirs(args.feature_dir, exist_ok=True)
    
    np.save(os.path.join(args.feature_dir, "vision_tower_features.npy"), vision_tower_features)
    np.save(os.path.join(args.feature_dir, "multimodal_projector_features.npy"), multimodal_projector_features)
    np.save(os.path.join(args.feature_dir, "language_model_features.npy"), language_model_features)
    print(f"Saved features to {args.feature_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--perspective-data", type=bool, default=True)
    parser.add_argument("--vitpose-coords", type=bool, default=False)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--feature-dir", type=str, default="../data/evals/perspective_taking/features")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
