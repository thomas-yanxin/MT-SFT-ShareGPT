# 根据分析结果抽取对应的数据

import json
import os

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analaysis import reward_distribution


def language_split(file_path):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    language_en = []
    language_zh = []
    language_other = []
    for item in data:
        if item["language"] == "en":
            language_en.append(item)
        elif item["language"] == "zh":
            language_zh.append(item)
        else:
            language_other.append(item)
    if len(language_zh) != 0:
        os.makedirs("data/zh", exist_ok=True)
        output_path_zh = os.path.join('data/zh', file_path.split(os.path.splitext(file_path)[-1])[0])
        with jsonlines.open(output_path_zh, mode="w") as writer:
            writer.write_all(language_zh)
    if len(language_en) != 0:
        os.makedirs("data/en", exist_ok=True)
        output_path_en = os.path.join('data/en', file_path.split(os.path.splitext(file_path)[-1])[0])
        with jsonlines.open(output_path_en, mode="w") as writer:
            writer.write_all(language_en)
    if len(language_other) != 0:
        os.makedirs("data/other", exist_ok=True)
        output_path_other = os.path.join('data/other', file_path.split(os.path.splitext(file_path)[-1])[0])
        with jsonlines.open(output_path_other, mode="w") as writer:
            writer.write_all(language_other)

    return [output_path_zh, output_path_en, output_path_other]


def reward_split(file_path, threshold):
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    
    # 根据reward的值按照绝对值大小进行排序
    data = sorted(data, key=lambda x: abs(x["reward"]), reverse=True)
    # 根据阈值划分数据
    reward_large = []
    reward_small = []
    for item in data:
        if abs(item["reward"]) > threshold:
            reward_large.append(item)
        else:
            reward_small.append(item)
    output_path_small = file_path.split(os.path.splitext(file_path)[-1])[0]+'_reward.jsonl'
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
    output_path_small = file_path.split(os.path.splitext(file_path)[-1])[0]+'_token_normal.jsonl'
    with jsonlines.open(output_path_small, mode="w") as writer:
        for item in tqdm(token_count_small):
            writer.write(item)
    
    output_path_long = file_path.split(os.path.splitext(file_path)[-1])[0]+'_token_count_long.jsonl'
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
        else:
            safety_high.append(item)
    output_path_safety = file_path.split(os.path.splitext(file_path)[-1])[0]+'_safety_high.jsonl'
    with jsonlines.open(output_path_safety, mode="w") as writer:
        for item in tqdm(safety_high):
            writer.write(item)

    return output_path_safety

def main(file_path):
    # 1. Split the data based on the language
    output_language_list = language_split(file_path)
    print(f"Data split based on language: {output_language_list}")

    # 2. Split the data based on the reward value
    threshold = 0.5

    for item in output_language_list:
        output_path_small = reward_split(item, threshold)
        print(f"Data split based on reward value: {output_path_small}")

        # 3. Split the data based on the token count
        token_num = 8192
        output_path_small, output_path_long = token_count_split(file_path, token_num)
        print(f"Data split based on token count: {output_path_small}, {output_path_long}")

        # 4. Split the data based on the safety value
        output_path_safety = safety_split(file_path)
        print(f"Data split based on safety value: {output_path_safety}")
    
    
if __name__ == "__main__":
    main("data.jsonl")


        

    

