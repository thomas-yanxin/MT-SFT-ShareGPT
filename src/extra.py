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
from tqdm import tqdm


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
        help="The task type. You can choose from ['all', 'language', 'reward', 'token_count', 'safety', 'quality', 'difficulty'], or use ',' to separate multiple tasks.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The threshold value for the reward.",
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
        default=["low"],
        help="The quality list to be removed. ['low', 'medium', 'high', 'very high']",
    )
    parser.add_argument(
        "--remove_difficulty_list",
        type=list,
        default=["very easy", "easy"],
        help="The difficulty list to be removed.",
    )

    return parser.parse_args()


args = get_args()


def language_split(file_path):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    language_en = []
    language_zh = []
    language_other = []
    for item in data:
        if item["language"] == "EN":
            language_en.append(item)
        elif item["language"] == "ZH":
            language_zh.append(item)
        else:
            language_other.append(item)
    if len(language_zh) != 0:
        os.makedirs("data/zh", exist_ok=True)
        output_path_zh = os.path.join(
            "data/zh", file_path.split(os.path.splitext(file_path)[-1])[0]
        )
        with jsonlines.open(output_path_zh, mode="w") as writer:
            writer.write_all(language_zh)
    if len(language_en) != 0:
        os.makedirs("data/en", exist_ok=True)
        output_path_en = os.path.join(
            "data/en", file_path.split(os.path.splitext(file_path)[-1])[0]
        )
        with jsonlines.open(output_path_en, mode="w") as writer:
            writer.write_all(language_en)
    if len(language_other) != 0:
        os.makedirs("data/other", exist_ok=True)
        output_path_other = os.path.join(
            "data/other", file_path.split(os.path.splitext(file_path)[-1])[0]
        )
        with jsonlines.open(output_path_other, mode="w") as writer:
            writer.write_all(language_other)

    return [output_path_zh, output_path_en, output_path_other]


def reward_split(file_path, threshold):
    with jsonlines.open(file_path) as reader:
        data = list(reader)

    # 根据reward的值按照绝对值大小进行排序
    data = sorted(data, key=lambda x: abs(x["rewards"]), reverse=True)
    # 根据阈值划分数据
    reward_large = []
    reward_small = []
    for item in data:
        if abs(item["rewards"]) > threshold:
            reward_large.append(item)
        else:
            reward_small.append(item)
    output_path_small = (
        file_path.split(os.path.splitext(file_path)[-1])[0] + "_reward.jsonl"
    )
    with jsonlines.open(output_path_small, mode="w") as writer:
        for item in tqdm(reward_small):
            writer.write(item)

    return output_path_small


def token_count_split(file_path, token_num=8192):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    token_count_large = []
    token_count_small = []

    for item in data:
        if item["token_count"] > token_num:
            token_count_large.append(item)
        else:
            token_count_small.append(item)
    output_path_small = (
        file_path.split(os.path.splitext(file_path)[-1])[0] + "_token_normal.jsonl"
    )
    with jsonlines.open(output_path_small, mode="w") as writer:
        for item in tqdm(token_count_small):
            writer.write(item)

    output_path_long = (
        file_path.split(os.path.splitext(file_path)[-1])[0] + "_token_count_long.jsonl"
    )
    with jsonlines.open(output_path_long, mode="w") as writer:
        for item in tqdm(token_count_large):
            writer.write(item)

    return output_path_small, output_path_long


def safety_split(file_path):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    safety_high = []
    for item in data:
        if "unsafe" in item["safety"]:
            pass
        elif item["safety"] == "":
            pass
        else:
            safety_high.append(item)
    output_path_safety = (
        file_path.split(os.path.splitext(file_path)[-1])[0] + "_safety_high.jsonl"
    )
    with jsonlines.open(output_path_safety, mode="w") as writer:
        for item in tqdm(safety_high):
            writer.write(item)

    return output_path_safety


def quality_split(file_path, remove_quality_list):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    quality_high = []
    for item in data:
        if item["quality"] in remove_quality_list:
            pass
        else:
            quality_high.append(item)
    output_path_quality = (
        file_path.split(os.path.splitext(file_path)[-1])[0] + "_quality_high.jsonl"
    )
    with jsonlines.open(output_path_quality, mode="w") as writer:
        for item in tqdm(quality_high):
            writer.write(item)

    return output_path_quality


def difficulty_split(file_path, remove_difficulty_list):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    difficulty_high = []
    for item in data:
        if item["difficulty"] in remove_difficulty_list:
            pass
        else:
            difficulty_high.append(item)
    output_path_difficulty = (
        file_path.split(os.path.splitext(file_path)[-1])[0] + "_difficulty_high.jsonl"
    )
    with jsonlines.open(output_path_difficulty, mode="w") as writer:
        for item in tqdm(difficulty_high):
            writer.write(item)

    return output_path_difficulty


def do_extra(
    file_path, task, threshold, token_num, remove_quality_list, remove_difficulty_list
):
    data_path = []
    if task == "all":
        language_list = language_split(file_path)
        for file_ in language_list:
            token_count_file = token_count_split(file_, token_num)
            safety_file = safety_split(token_count_file)
            quality_file = quality_split(safety_file, remove_quality_list)
            difficulty_file = difficulty_split(quality_file, remove_difficulty_list)
            reward_file = reward_split(difficulty_file, threshold)
    elif task == "language":
        language_list = language_split(file_path)
        data_path.extend(language_list)
    elif task == "reward":
        reward_file = reward_split(file_path, threshold)
        data_path.append(reward_file)

    elif task == "token_count":
        token_count_file = token_count_split(file_path, token_num)
        data_path.extend(token_count_file)

    elif task == "safety":
        safety_file = safety_split(file_path)
        data_path.append(safety_file)
    elif task == "quality":
        quality_file = quality_split(file_path, remove_quality_list)
        data_path.append(quality_file)
    elif task == "difficulty":
        difficulty_file = difficulty_split(file_path, remove_difficulty_list)
        data_path.append(difficulty_file)

    else:
        print("The task is not supported.")

    return data_path


def main():
    # 判断data_path是文件夹还是文件
    if os.path.isdir(args.file_path):
        file_list = os.listdir(args.file_path)
        for file in file_list:
            if file.endswith(".jsonl"):
                args.file_path = os.path.join(args.file_path, file)
                if "," in args.task:
                    task_list = args.task.split(",")
                    for n, task in enumerate(task_list):
                        if n == 0:
                            data_path = do_extra(
                                args.file_path,
                                task,
                                args.threshold,
                                args.token_num,
                                args.remove_quality_list,
                                args.remove_difficulty_list,
                            )
                        else:
                            for i in range(len(data_path)):
                                data_path[i] = do_extra(
                                    data_path[i],
                                    task,
                                    args.threshold,
                                    args.token_num,
                                    args.remove_quality_list,
                                    args.remove_difficulty_list,
                                )
                else:
                    do_extra(
                        args.file_path,
                        args.task,
                        args.threshold,
                        args.token_num,
                        args.remove_quality_list,
                        args.remove_difficulty_list,
                    )

    elif os.path.isfile(args.file_path):
        if "," in args.task:
            task_list = args.task.split(",")
            for n, task in enumerate(task_list):
                if n == 0:
                    data_path = do_extra(
                        args.file_path,
                        task,
                        args.threshold,
                        args.token_num,
                        args.remove_quality_list,
                        args.remove_difficulty_list,
                    )
                else:
                    for i in range(len(data_path)):
                        data_path[i] = do_extra(
                            data_path[i],
                            task,
                            args.threshold,
                            args.token_num,
                            args.remove_quality_list,
                            args.remove_difficulty_list,
                        )

        else:
            do_extra(
                args.file_path,
                args.task,
                args.threshold,
                args.token_num,
                args.remove_quality_list,
                args.remove_difficulty_list,
            )

    else:
        print("The file path is not valid.")


if __name__ == "__main__":

    main()
