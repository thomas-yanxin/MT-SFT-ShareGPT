import json
import os
import sys

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline


def difficulty_distribution(data):
    difficulty_list = [item["difficulty"] for item in data]
    # 替换列表中的None值
    difficulty_list = [item if item is not None else "None" for item in difficulty_list]
    difficulty_dict = {}
    for item in tqdm(difficulty_list):
        if item in difficulty_dict:
            difficulty_dict[item] += 1
        else:
            difficulty_dict[item] = 1
    return difficulty_dict


def quality_distribution(data):
    quality_list = [item["quality"] for item in data]
    quality_list = [item if item is not None else "None" for item in quality_list]
    quality_dict = {}
    for item in tqdm(quality_list):
        if item in quality_dict:
            quality_dict[item] += 1
        else:
            quality_dict[item] = 1
    return quality_dict


def classification_distribution(data):
    classification_list = [item["classification"] for item in data]
    classification_list = [item if item is not None else "None" for item in classification_list]
    classification_dict = {}
    for item in tqdm(classification_list):
        if item in classification_dict:
            classification_dict[item] += 1
        else:
            classification_dict[item] = 1
    return classification_dict

def safety_distribution(data):
    safety_list = [item["safety"] for item in data]
    safety_dict = {}
    for item in tqdm(safety_list):
        if item in safety_dict:
            safety_dict[item] += 1
        else:
            safety_dict[item] = 1
    return safety_dict

def reward_distribution(data):
    reward_list = [item["reward"] for item in data]
    reward_diff_list = [abs(item) for item in reward_list]
    reward_diff_dict = {}
    # 平均奖励值
    reward_diff_dict["average"] = sum(reward_diff_list) / len(reward_diff_list)
    # 最大奖励值
    reward_diff_dict["max"] = max(reward_diff_list)
    # 最小奖励值
    reward_diff_dict["min"] = min(reward_diff_list)
    # 奖励值分布
    
    for item in reward_diff_list:
        reward_diff_dict[round(float(item))] += 1

    return reward_diff_dict

def language_distribution(data):
    language_list = [item["language"] for item in data]
    language_dict = {}
    for item in tqdm(language_list):
        if item == "EN":
            language_dict["EN"] = language_dict.get("EN", 0) + 1
        elif item == "ZH":
            language_dict["ZH"] = language_dict.get("ZH", 0) + 1
        else:
            language_dict["OTHER"] = language_dict.get("OTHER", 0) + 1
    return language_dict


def token_count_distribution(data):
    token_count_list = [item["token_count"] for item in data]
    token_count_dict = {}
    token_cound_range = [0, 128, 256, 512, 1024, 2048, 4096, 8192]
    for i in range(len(token_cound_range) - 1):
        token_count_dict[f"{token_cound_range[i]}-{token_cound_range[i + 1]}"] = 0
    token_count_dict[f"{token_cound_range[-1]}-"] = 0
    for item in tqdm(token_count_list):
        for i in range(len(token_cound_range) - 1):
            if token_cound_range[i] <= item < token_cound_range[i + 1]:
                token_count_dict[f"{token_cound_range[i]}-{token_cound_range[i + 1]}"] += 1
                break
        if item >= token_cound_range[-1]:
            token_count_dict[f"{token_cound_range[-1]}-"] += 1
    return token_count_dict


## 分析数据各个维度的分布
def analyze_data_distribution(data_path):
    with jsonlines.open(data_path) as reader:
        data = list(reader)
    print(f"Number of data samples: {len(data)}")
    print(f"Example data: {data[0]}")
    key_list = [key for key in data[0].keys()]
    # ["difficulty", "quality", "classification", "safety", "reward", "language", "token_count"]
    # 分析并画图，存进./images文件夹
    if not os.path.exists("./images"):
        os.mkdir("./images")
    for key in key_list:
        if key == "difficulty":
            difficulty_dict = difficulty_distribution(data)
            print(f"Difficulty distribution: {difficulty_dict}")
            plt.bar(difficulty_dict.keys(), difficulty_dict.values())
            plt.xlabel("Difficulty")
            plt.ylabel("Count")
            plt.title("Difficulty Distribution")
            plt.savefig(f"./images/difficulty_distribution.png")
            plt.show()
        elif key == "quality":
            quality_dict = quality_distribution(data)
            print(f"Quality distribution: {quality_dict}")
            plt.bar(quality_dict.keys(), quality_dict.values())
            plt.xlabel("Quality")
            plt.ylabel("Count")
            plt.title("Quality Distribution")
            plt.savefig(f"./images/quality_distribution.png")
            plt.show()
        elif key == "classification":
            classification_dict = classification_distribution(data)
            print(f"Classification distribution: {classification_dict}")
            plt.bar(classification_dict.keys(), classification_dict.values())
            plt.xlabel("Classification")
            plt.ylabel("Count")
            plt.title("Classification Distribution")
            plt.savefig(f"./images/classification_distribution.png")
            plt.show()
        elif key == "safety":
            safety_dict = safety_distribution(data)
            print(f"Safety distribution: {safety_dict}")
            plt.bar(safety_dict.keys(), safety_dict.values())
            plt.xlabel("Safety")
            plt.ylabel("Count")
            plt.title("Safety Distribution")
            plt.savefig(f"./images/safety_distribution.png")
            plt.show()
        elif key == "reward":
            reward_dict = reward_distribution(data)
            print(f"Reward distribution: {reward_dict}")
            plt.bar(reward_dict.keys(), reward_dict.values())
            plt.xlabel("Reward")
            plt.ylabel("Count")
            plt.title("Reward Distribution")
            plt.savefig(f"./images/reward_distribution.png")
            plt.show()
        elif key == "language":
            language_dict = language_distribution(data)
            print(f"Language distribution: {language_dict}")
            plt.bar(language_dict.keys(), language_dict.values())
            plt.xlabel("Language")
            plt.ylabel("Count")
            plt.title("Language Distribution")
            plt.savefig(f"./images/language_distribution.png")
            plt.show()
        elif key == "token_count":
            token_count_dict = token_count_distribution(data)
            print(f"Token count distribution: {token_count_dict}")
            plt.bar(token_count_dict.keys(), token_count_dict.values())
            plt.xlabel("Token Count")
            plt.ylabel("Count")
            plt.title("Token Count Distribution")


if __name__ == "__main__":
    data_path = sys.argv[1]
    analyze_data_distribution(data_path)