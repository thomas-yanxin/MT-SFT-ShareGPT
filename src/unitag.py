import argparse
import gc
import json
import os

import jsonlines
import torch
from distilabel.llms import OpenAILLM, vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, MinHashDedup
from distilabel.steps.tasks import MagpieGenerator
from lingua import Language, LanguageDetectorBuilder
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

from util import (Classification, Difficulty, Quality, conversations_mapping,
                  input_classification, input_difficulty_rating,
                  input_quality_rating, load_jsonl_to_list, refined_result)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
detector = LanguageDetectorBuilder.from_all_languages().build()


def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="SFT Datasets Tagging Manager.")
    parser.add_argument(
        "--tag_mission",
        type=str,
        default="quality",
        help="The tagging mission.",
        choices=[
            "difficulty",
            "quality",
            "classification",
            "safety",
            "reward",
            "language",
            "token_count",
            "refined",
            "all",
        ],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Tag Model.",
    )
    parser.add_argument(
        "--guard_model_path",
        type=str,
        default="meta-llama/Llama-Guard-3-8B",
        help="Guard Model.",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="internlm/internlm2-7b-reward",
        help="Reward Model.",
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="Input dataset file name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of samples per batch. Online <100, Offline <200.",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="jsonl",
        choices=["jsonl"],
        help="Save the generated responses as a what kind of file",
    )

    # vllm Configs
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"]
    )
    parser.add_argument(
        "--quantization", type=str, default="fp8", choices=["fp8", "awq", "gptq", None]
    )
    parser.add_argument(
        "--kv_cache_dtype", type=str, default="auto", choices=["auto", "fp8"]
    )
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)

    # Tagging Generation Configs
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    return parser.parse_args()


args = get_args()


def set_schema(tag_mission):
    schema_map = {
        "difficulty": (Difficulty, input_difficulty_rating, "medium", "difficulty"),
        "classification": (Classification, input_classification, "Others", "primary_tag"),
        "quality": (Quality, input_quality_rating, "average", "input_quality")
    }
    
    return schema_map.get(tag_mission, (None, None, None, None))


def gen_llm(device, model, tag_mission, max_tokens, temperature, repetition_penalty, gpu_memory_utilization=0.95):
    cuda_devices = [int(d) for d in device.split(",")] if "," in device else [int(device)]
    
    return vLLM(
        cuda_devices=cuda_devices,
        model=model,
        structured_output={"format": "json", "schema": set_schema(tag_mission)[0]},
        extra_kwargs={
            "tensor_parallel_size": len(cuda_devices),
            "distributed_executor_backend": "ray",
        },
        generation_kwargs={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "gpu_memory_utilization": gpu_memory_utilization,
        },
    )


def generate_tags(
    device,
    model,
    tag_mission,
    max_tokens,
    temperature,
    repetition_penalty,
    input_file,
    batch_size,
    save_as,
    gpu_memory_utilization=0.95,
):
    print(f"正在生成{tag_mission}标签...")

    llm = gen_llm(
        device, model, tag_mission, max_tokens, temperature, 
        repetition_penalty, gpu_memory_utilization
    )
    llm.load()

    output_file = f"{os.path.splitext(input_file)[0]}_{tag_mission}.{save_as}"
    schema = set_schema(tag_mission)
    input_rating, default_value, result_key = schema[1], schema[2], schema[3]

    with jsonlines.open(output_file, mode='w') as writer:
        data_list = load_jsonl_to_list(input_file)
        for batch in tqdm([data_list[i:i+batch_size] for i in range(0, len(data_list), batch_size)]):
            prompts = [
                [{"role": "user", "content": input_rating(item["conversations"][1 if item["conversations"][0]["from"] == "system" else 0]["value"])}]
                for item in batch
            ]
            
            results = llm.generate(prompts)
            
            for item, result in zip(batch, results):
                try:
                    item[tag_mission] = json.loads(result[0])[result_key]
                except:
                    item[tag_mission] = default_value
                writer.write(item)

    return output_file


def token_count(input_file, model):
    print("计算token数量...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    output_file = f"{os.path.splitext(input_file)[0]}_token_count.jsonl"

    with jsonlines.open(output_file, mode='w') as writer:
        for item in tqdm(load_jsonl_to_list(input_file)):
            text = tokenizer.apply_chat_template(
                conversations_mapping(item["conversations"]),
                tokenize=False,
                add_generation_prompt=True
            )
            item["token_count"] = len(tokenizer.encode(text))
            writer.write(item)

    return output_file


def language_detection_turns(input_file):
    print("检测语言和对话轮次...")
    output_file = f"{os.path.splitext(input_file)[0]}_language_turns.jsonl"
    
    with jsonlines.open(output_file, mode='w') as writer:
        for item in tqdm(load_jsonl_to_list(input_file)):
            text = ''.join(conv['value'] for conv in item['conversations'])
            turns = (len(item['conversations']) - (item['conversations'][0]['from'] == 'system')) // 2
            
            try:
                lang = detector.detect_language_of(text).iso_code_639_1.name
            except:
                lang = None
            
            item.update({'language': lang, 'turns': turns})
            writer.write(item)
    
    return output_file


def safety_get_completion(prompts, llm):
    sampling_params = SamplingParams()
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def safety_tag(model, device, input_file, batch_size, save_as):
    print("生成安全标签...")
    cuda_devices = [int(d) for d in device.split(",")] if "," in device else [int(device)]
    
    safety_llm = LLM(
        model=model,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=len(cuda_devices),
        enable_prefix_caching=True,
        max_model_len=4096,
    )
    safety_tokenizer = AutoTokenizer.from_pretrained(model)

    output_file = f"{os.path.splitext(input_file)[0]}_safety.{save_as}"
    
    with jsonlines.open(output_file, mode='w') as writer:
        data_list = load_jsonl_to_list(input_file)
        for batch in tqdm([data_list[i:i+batch_size] for i in range(0, len(data_list), batch_size)]):
            prompts = [safety_tokenizer.apply_chat_template(conversations_mapping(item["conversations"]), tokenize=False) for item in batch]
            results = safety_get_completion(prompts, llm=safety_llm)
            
            for item, result in zip(batch, results):
                item["safety"] = "unsafe" if "unsafe" in result else "safe"
                writer.write(item)

    cleanup(safety_llm)
    return output_file

def cleanup(llm):
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def reward_tag(model, device, input_file, batch_size, save_as):
    print("生成奖励标签...")
    cuda_devices = [int(d) for d in device.split(",")] if "," in device else [int(device)]

    reward_tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    reward_model = AutoModel.from_pretrained(
        model,
        device_map=f"cuda:{cuda_devices[0]}",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    output_file = f"{os.path.splitext(input_file)[0]}_reward.{save_as}"
    
    with jsonlines.open(output_file, mode='w') as writer:
        data_list = load_jsonl_to_list(input_file)
        
        for item in tqdm(data_list):
            token_count = item.get("token_count", 0)
            if not token_count:
                text = reward_tokenizer.apply_chat_template(
                    conversations_mapping(item["conversations"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                token_count = len(reward_tokenizer.encode(text))
            
            item["reward"] = 0 if token_count > 8192 else reward_model.get_score(
                reward_tokenizer, conversations_mapping(item["conversations"])
            )
            writer.write(item)

    return output_file



def main():
    tag_functions = {
        "difficulty": generate_tags,
        "classification": generate_tags,
        "quality": generate_tags,
        "language": language_detection_turns,
        "safety": safety_tag,
        "reward": reward_tag,
        "token_count": token_count,
        "refined": refined_result
    }

    if args.tag_mission in tag_functions:
        func = tag_functions[args.tag_mission]
        if args.tag_mission in ["difficulty", "classification", "quality"]:
            func(args.device, args.model_path, args.tag_mission, args.max_tokens,
                 args.temperature, args.repetition_penalty, args.input_file,
                 args.batch_size, args.save_as)
        elif args.tag_mission in ["safety", "reward"]:
            model_path = args.guard_model_path if args.tag_mission == "safety" else args.reward_model_path
            func(model_path, args.device, args.input_file, args.batch_size, args.save_as)
        elif args.tag_mission == "token_count":
            func(args.input_file, args.model_path)
        else:
            func(args.input_file, args.save_as)
    else:
        print("无效的标签任务。")


if __name__ == "__main__":
    main()
