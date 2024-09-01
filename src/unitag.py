import argparse
import gc
import json
import os

import jsonlines
import torch
from distilabel.llms import OpenAILLM, vLLM
from distilabel.steps.tasks import MagpieGenerator
from lingua import Language, LanguageDetectorBuilder
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

from prompt import (Classification, Difficulty, Quality, input_classification,
                    input_difficulty_rating, input_quality_rating)

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
    if tag_mission in ["difficulty", "classification", "quality"]:
        if tag_mission == "difficulty":
            schema = Difficulty
            input_rating = input_difficulty_rating
            error_message = "medium"
            output_message = "difficulty"
        elif tag_mission == "classification":
            schema = Classification
            input_rating = input_classification
            error_message = "Others"
            output_message = "primary_tag"
        elif tag_mission == "quality":
            schema = Quality
            input_rating = input_quality_rating
            error_message = "average"
            output_message = "input_quality"

    return schema, input_rating, error_message, output_message


def load_jsonl_to_list(jsonl_file_path):
    data_list = []
    with open(jsonl_file_path, "r", encoding="utf8") as file:
        for line in jsonlines.Reader(file):
            data_list.append(line)
    return data_list


def gen_llm(
    device,
    model,
    tag_mission,
    max_tokens,
    temperature,
    repetition_penalty,
    gpu_memory_utilization=0.95,
):
    if "," in device:
        cuda_devices = [int(d) for d in device.split(",")]
    else:
        cuda_devices = [int(device)]

    tensor_parallel_size = len(cuda_devices)

    llm = vLLM(
        cuda_devices=cuda_devices,
        model=model,
        structured_output={"format": "json", "schema": set_schema(tag_mission)[0]},
        extra_kwargs={
            "tensor_parallel_size": tensor_parallel_size,
            "distributed_executor_backend": "ray",
        },
        generation_kwargs={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "gpu_memory_utilization": gpu_memory_utilization,
        },
    )
    return llm


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
    print("Generating {tag_mission} tags...".format(tag_mission=tag_mission))

    llm = gen_llm(
        device,
        model,
        tag_mission,
        max_tokens,
        temperature,
        repetition_penalty,
        gpu_memory_utilization,
    )
    llm.load()

    with open(
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_{tag_mission}.{save_as}",
        "a+",
        encoding="utf8",
    ) as out:

        input_rating = set_schema(tag_mission)[1]
        data_list = load_jsonl_to_list(input_file)
        for i in tqdm(range(0, len(data_list), batch_size)):
            all_batch = data_list[i : i + batch_size]

            after_batch = []
            for batch in all_batch:
                if "system" in batch["conversations"][0]["from"]:
                    after_batch.append(
                        [
                            {
                                "role": "user",
                                "content": input_rating(
                                    batch["conversations"][1]["value"]
                                ),
                            },
                        ]
                    )
                else:
                    after_batch.append(
                        [
                            {
                                "role": "user",
                                "content": input_rating(
                                    batch["conversations"][0]["value"]
                                ),
                            },
                        ]
                    )

            result = llm.generate(after_batch)
            for num, res in enumerate(zip(all_batch, result)):
                try:
                    data = json.loads(res[1][0])
                    all_batch[num][tag_mission] = data[set_schema(tag_mission)[3]]
                except:
                    all_batch[num][tag_mission] = set_schema(tag_mission)[2]
                jsonlines.Writer(out).write(all_batch[num])

    output_file = f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_{tag_mission}.{save_as}"

    return output_file


def token_count(input_file, model):
    print("Calculating token count...")
    tokenizer = AutoTokenizer.from_pretrained(model)

    with open(
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_token_count.jsonl",
        "a+",
        encoding="utf8",
    ) as out:
        data_list = load_jsonl_to_list(input_file)
        for i, n in tqdm(enumerate(data_list)):

            text_ = tokenizer.apply_chat_template(
                conversations_mapping(n["conversations"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            # 计算text的token数量
            token_count = len(tokenizer.encode(text_))
            n["token_count"] = token_count
            jsonlines.Writer(out).write(n)

    output_file = (
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_token_count.jsonl"
    )

    return output_file


def conversations_mapping(conversations):
    # 将ShareGPT转chatml格式
    chatml = []
    for i in conversations:
        if "system" in i["from"]:
            chatml.append({"role": "system", "content": i["value"]})
        else:
            if "human" in i["from"]:
                chatml.append({"role": "user", "content": i["value"]})
            elif "gpt" in i["from"]:
                chatml.append({"role": "assistant", "content": i["value"]})
    return chatml


def language_detection_turns(input_file):
    print("Detecting language and turns...")

    with open(
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_language_turns.jsonl",
        "a+",
        encoding="utf8",
    ) as out:
        data_list = load_jsonl_to_list(input_file)
        for i, n in tqdm(enumerate(data_list)):
            text = ""
            for i in n["conversations"]:
                text += i["value"]
            if n["conversations"][0]["from"] == "system":
                turns = round((len(n["conversations"][1:])) / 2 )
            else:
                turns = round((len(n["conversations"])) / 2 ) 
            try:
                lang = detector.detect_language_of(text).iso_code_639_1.name
            except:
                lang = None

            n["language"] = lang
            n["turns"] = turns

            jsonlines.Writer(out).write(n)

    output_file = (
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_language_turns.jsonl"
    )
    return output_file


def safety_get_completion(prompts, llm):
    sampling_params = SamplingParams()
    outputs = llm.generate(prompts, sampling_params)
    responses = []
    for output in outputs:
        response = output.outputs[0].text
        responses.append(response)
    return responses


def safety_tag(model, device, input_file, batch_size, save_as):
    print("Generating safety tags...")
    if "," in device:
        cuda_devices = [int(d) for d in device.split(",")]
    else:
        cuda_devices = [int(device)]
    tensor_parallel_size = len(cuda_devices)

    safety_llm = LLM(
        model=model,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=True,
        max_model_len=4096,
    )
    safety_tokenizer = AutoTokenizer.from_pretrained(model)

    with open(
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_safety.{save_as}",
        "a+",
        encoding="utf8",
    ) as out:

        all_batch = []
        data_list = load_jsonl_to_list(input_file)
        for i in tqdm(range(0, len(data_list), batch_size)):
            all_batch = data_list[i : i + batch_size]
            prompts_safety = []
            for item in all_batch:
                text_safety = safety_tokenizer.apply_chat_template(
                    conversations_mapping(item["conversations"]), tokenize=False
                )
                prompts_safety.append(text_safety)

            result = safety_get_completion(prompts_safety, llm=safety_llm)
            refined_result = []
            for res in result:
                if "unsafe" in res:
                    refined_result.append("unsafe")
                else:
                    refined_result.append("safe")

            for num, res in enumerate(zip(all_batch, refined_result)):
                all_batch[num]["safety"] = res[1]
                jsonlines.Writer(out).write(all_batch[num])

    output_file = (
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_safety.{save_as}"
    )
    destroy_model_parallel()
    destroy_distributed_environment()
    del safety_llm.llm_engine.model_executor
    del safety_llm
    gc.collect()
    torch.cuda.empty_cache()
    return output_file


def reward_tag(model, device, input_file, batch_size, save_as):
    print("Generating reward tags...")
    if "," in device:
        cuda_devices = [int(d) for d in device.split(",")]
    else:
        cuda_devices = [int(device)]

    reward_tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    reward_model = AutoModel.from_pretrained(
        model,
        device_map="cuda:{device}".format(device=cuda_devices[0]),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    with open(
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_reward.{save_as}",
        "a+",
        encoding="utf8",
    ) as out:

        data_list = load_jsonl_to_list(input_file)

        batch_size = 1
        for n in tqdm(data_list):

            token_count = 0

            if "token_count" in n:
                token_count = n["token_count"]
            else:
                text_ = reward_tokenizer.apply_chat_template(
                    conversations_mapping(n["conversations"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                token_count = len(reward_tokenizer.encode(text_))

            if token_count > 8192:
                n["reward"] = 0

                jsonlines.Writer(out).write(n)
            else:
                result = reward_model.get_scores(
                    reward_tokenizer, conversations_mapping(n["conversations"])
                )
                n["reward"] = result
                jsonlines.Writer(out).write(n)

    output_file = (
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_reward.{save_as}"
    )
    return output_file


def refined_result(input_file, save_as):
    with open(
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_refined.{save_as}",
        "a+",
        encoding="utf8",
    ) as out:
        with open(input_file, "r", encoding="utf8") as file:
            key_list = [
                "id",
                "conversations",
                "difficulty",
                "classification",
                "quality",
                "safety",
                "rewards",
                "language",
                "turns",
                "token_count",
                "source",
            ]
            for line in jsonlines.Reader(file):
                refined = {}
                for key in key_list:
                    if key in line:
                        if key == "difficulty":
                            if line[key] in ["very easy", "easy", "medium", "hard", "very hard"]:
                                refined[key] = line[key]
                            else:
                                refined[key] = "medium"
                        elif key == "classification":    
                            if line[key] in ["Information seeking", "Reasoning", "Planning", "Editing", "Coding & Debugging", "Math", "Role playing", "Data analysis", "Creative writing", "Advice seeking", "Brainstorming", "Others"]:
                                refined[key] = line[key]
                            else:
                                refined[key] = "Others"
                        elif key == "quality":
                            # [very poor/poor/average/good/excellent]
                            if line[key] in ["very poor", "poor", "average", "good", "excellent"]:
                                refined[key] = line[key]
                            else:
                                refined[key] = "average"
                        elif key == "safety":
                            if line[key] in ["unsafe", "safe"]:
                                refined[key] = line[key]
                            else:
                                refined[key] = "unsafe"
                        elif key == "rewards":
                            refined[key] = float(line[key])
                        else:                        
                            refined[key] = line[key]
                    else:
                        if key == "difficulty":
                            refined[key] = "medium"
                        elif key == "classification":
                            refined[key] = "Others"
                        elif key == "quality":
                            refined[key] = "average"
                        elif key == "safety":
                            refined[key] = "safe"
                        elif key == "rewards":
                            refined[key] = 0
                        elif key == "language":
                            refined[key] = None
                        elif key == "turns":
                            refined[key] = 1
                        elif key == "token_count":
                            refined[key] = 0
                        elif key == "source":
                            refined[key] = "None"
                jsonlines.Writer(out).write(refined)

    output_file = (
        f"{input_file.split(os.path.splitext(input_file)[-1])[0]}_refined.{save_as}"
    )
    return output_file


def main():
    if args.tag_mission in ["difficulty", "classification", "quality"]:
        generate_tags(
            args.device,
            args.model_path,
            args.tag_mission,
            args.max_tokens,
            args.temperature,
            args.repetition_penalty,
            args.input_file,
            args.batch_size,
            args.save_as,
        )
    elif args.tag_mission == "language":
        language_detection_turns(args.input_file)
    elif args.tag_mission == "safety":
        safety_tag(
            args.guard_model_path,
            args.device,
            args.input_file,
            args.batch_size,
            args.save_as,
        )
    elif args.tag_mission == "reward":
        reward_tag(
            args.reward_model_path,
            args.device,
            args.input_file,
            args.batch_size,
            args.save_as,
        )
    elif args.tag_mission == "token_count":
        token_count(args.input_file, args.model_path)
    elif args.tag_mission == "refined":
        refined_result(args.input_file, args.save_as)
    else:
        print("Invalid tag mission.")


if __name__ == "__main__":
    main()
