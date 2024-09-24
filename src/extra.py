# 根据分析结果抽取对应的数据

import argparse
import gc
import json
import os
import sys

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, MinHashDedup
from tqdm import tqdm

from util import load_jsonl_to_list


def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(
        description="Extract the data based on the given task."
    )
    parser.add_argument("--file_path", type=str, help="The path of the input file.")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="The task type. You can choose from ['all', 'language', 'rewards', 'token_count', 'safety', 'quality', 'difficulty', 'deduplication'], or use ',' to separate multiple tasks.",
    )
    parser.add_argument(
        "--rewards_threshold",
        type=float,
        default=1,
        help="The threshold value for the reward.",
    )
    parser.add_argument(
        "--deduplication_threshold",
        type=float,
        default=0.9,
        help="The threshold value for the deduplication.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="The batch size for the deduplication.",
    )
    parser.add_argument(
        "--token_num",
        type=int,
        default=8192,
        help="The token number for the token count.",
    )
    parser.add_argument(
        "--remove_quality_list",
        type=list,
        default="very poor",
        help="The quality list to be removed. 'very poor', 'poor', 'low', 'medium', 'high', 'very high'.",
    )
    parser.add_argument(
        "--remove_difficulty_list",
        type=list,
        default="very easy,easy",
        help="The difficulty list to be removed. 'very easy', 'easy', 'medium', 'hard', 'very hard'.",
    )
    

    return parser.parse_args()


args = get_args()


def language_split(file_path):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    
    languages = {"EN": [], "ZH": [], "other": []}
    for item in data:
        lang = item["language"]
        languages[lang if lang in ["EN", "ZH"] else "other"].append(item)
    
    output_paths = []
    for lang, items in languages.items():
        if items:
            output_dir = f"data/{lang.lower()}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0])
            with jsonlines.open(output_path, mode="w") as writer:
                writer.write_all(items)
            output_paths.append(output_path)
    
    return output_paths


def reward_split(file_path, threshold):
    with jsonlines.open(file_path) as reader:
        data = sorted(reader, key=lambda x: abs(float(x["rewards"])), reverse=True)
    
    reward_small = [item for item in data if abs(float(item["rewards"])) <= threshold]
    
    output_path = f"{os.path.splitext(file_path)[0]}_rewards.jsonl"
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(tqdm(reward_small))
    
    return output_path


def token_count_split(file_path, token_num=8192):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    
    token_count_large = [item for item in data if item["token_count"] > token_num]
    token_count_small = [item for item in data if item["token_count"] <= token_num]
    
    base_path = os.path.splitext(file_path)[0]
    output_paths = {
        "small": f"{base_path}_token_normal.jsonl",
        "large": f"{base_path}_token_count_long.jsonl"
    }
    
    for key, items in zip(["small", "large"], [token_count_small, token_count_large]):
        with jsonlines.open(output_paths[key], mode="w") as writer:
            writer.write_all(tqdm(items))
    
    return output_paths["small"], output_paths["large"]


def safety_split(file_path):
    with jsonlines.open(file_path) as reader:
        safety_high = [item for item in reader if item["safety"] not in ["unsafe", ""]]
    
    output_path_safety = f"{os.path.splitext(file_path)[0]}_safety_high.jsonl"
    
    with jsonlines.open(output_path_safety, mode="w") as writer:
        writer.write_all(tqdm(safety_high))

    return output_path_safety


def quality_split(file_path, remove_quality_list):
    remove_quality_list = set(remove_quality_list.split(","))
    output_path_quality = f"{os.path.splitext(file_path)[0]}_quality_high.jsonl"
    
    with jsonlines.open(file_path) as reader, jsonlines.open(output_path_quality, mode="w") as writer:
        for item in tqdm(reader):
            if item["quality"] not in remove_quality_list:
                writer.write(item)

    return output_path_quality


def difficulty_split(file_path, remove_difficulty_list):
    remove_difficulty_list = set(remove_difficulty_list.split(","))
    output_path = f"{os.path.splitext(file_path)[0]}_difficulty_high.jsonl"
    
    with jsonlines.open(file_path) as reader, jsonlines.open(output_path, mode="w") as writer:
        for item in tqdm(reader):
            if item["difficulty"] not in remove_difficulty_list:
                writer.write(item)

    return output_path


def deduplication(file_path, threshold=0.9, batch_size=500):
    output_path = f"{os.path.splitext(file_path)[0]}_deduplication.jsonl"
    
    with jsonlines.open(output_path, mode='w') as writer:
        data_list = load_jsonl_to_list(file_path)
        
        with Pipeline() as pipeline:
            data = LoadDataFromDicts(data=data_list, batch_size=batch_size)
            minhash_dedup = MinHashDedup(
                tokenizer="words",
                threshold=threshold,
                storage="dict",
            )
            data >> minhash_dedup
        
        distiset = pipeline.run(use_cache=False)
        ds_dedup = distiset["default"]["train"].filter(lambda x: x["keep_row_after_minhash_filtering"])
        
        for item in ds_dedup:
            item.pop('text')
            writer.write(item)
    
    return output_path
    


def do_extra(file_path, task, rewards_threshold, token_num, remove_quality_list, remove_difficulty_list, deduplication_threshold, batch_size):
    task_functions = {
        "language": lambda: language_split(file_path),
        "rewards": lambda: [reward_split(file_path, rewards_threshold)],
        "token_count": lambda: token_count_split(file_path, token_num),
        "safety": lambda: [safety_split(file_path)],
        "quality": lambda: [quality_split(file_path, remove_quality_list)],
        "difficulty": lambda: [difficulty_split(file_path, remove_difficulty_list)],
        "deduplication": lambda: [deduplication(file_path, deduplication_threshold, batch_size)]
    }

    if task == "all":
        return [deduplication(
            reward_split(
                difficulty_split(
                    quality_split(
                        safety_split(
                            token_count_split(file_, token_num)
                        ), remove_quality_list
                    ), remove_difficulty_list
                ), rewards_threshold
            ), deduplication_threshold, batch_size
        ) for file_ in language_split(file_path)]
    
    if task in task_functions:
        return task_functions[task]()
    
    print("不支持的任务。")
    return []


def main():
    # 判断data_path是文件夹还是文件
    if os.path.isdir(args.file_path):
        file_list = [f for f in os.listdir(args.file_path) if f.endswith(".jsonl")]
    elif os.path.isfile(args.file_path):
        file_list = [args.file_path]
    else:
        print("文件路径无效。")
        return

    for file in file_list:
        file_path = os.path.join(args.file_path, file) if os.path.isdir(args.file_path) else file
        tasks = args.task.split(",") if "," in args.task else [args.task]
        
        result = [file_path]
        for task in tasks:
            result = [
                path for file_path in result
                for path in do_extra(
                    file_path,
                    task,
                    args.rewards_threshold,
                    args.token_num,
                    args.remove_quality_list,
                    args.remove_difficulty_list,
                    args.deduplication_threshold,
                    args.batch_size,
                )
            ]


if __name__ == "__main__":

    main()
