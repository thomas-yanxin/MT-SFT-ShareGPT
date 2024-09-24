import json
import os
import sys

import jsonlines
import matplotlib.pyplot as plt
from tqdm import tqdm


def difficulty_distribution(data):
    difficulty_list = [item.get("difficulty", "None") for item in data]
    return {difficulty: difficulty_list.count(difficulty) for difficulty in set(difficulty_list)}


def quality_distribution(data):
    quality_list = [item.get("quality", "None") for item in data]
    return {quality: quality_list.count(quality) for quality in set(quality_list)}


def classification_distribution(data):
    classification_list = [item.get("classification", "None") for item in data]
    return {cls: classification_list.count(cls) for cls in set(classification_list)}


def safety_distribution(data):
    safety_list = [item["safety"] for item in data]
    return {safety: safety_list.count(safety) for safety in set(safety_list)}


def reward_distribution(data):
    rewards = [abs(float(item["rewards"])) for item in data]
    return {
        "average": sum(rewards) / len(rewards),
        "max": max(rewards),
        "min": min(rewards)
    }


def language_distribution(data):
    languages = [item["language"] for item in data]
    return {
        "EN": languages.count("EN"),
        "ZH": languages.count("ZH"),
        "OTHER": len(languages) - languages.count("EN") - languages.count("ZH")
    }


def token_count_distribution(data):
    ranges = [0, 128, 256, 512, 1024, 2048, 4096, 8192]
    distribution = {f"{ranges[i]}-{ranges[i+1]}": 0 for i in range(len(ranges)-1)}
    distribution[f"{ranges[-1]}-"] = 0

    for count in tqdm([item["token_count"] for item in data]):
        for i, upper in enumerate(ranges[1:], 1):
            if count < upper:
                distribution[f"{ranges[i-1]}-{upper}"] += 1
                break
        else:
            distribution[f"{ranges[-1]}-"] += 1

    return distribution


## 分析数据各个维度的分布
def analyze_data_distribution(data_path):
    # 读取数据
    data = []
    if os.path.isdir(data_path):
        for file in os.listdir(data_path):
            with jsonlines.open(os.path.join(data_path, file)) as reader:
                data.extend(list(reader))
    else:
        with jsonlines.open(data_path) as reader:
            data = list(reader)
    
    print(f"数据样本数量: {len(data)}")
    print(f"示例数据: {data[0]}")
    
    # 创建图像保存目录
    os.makedirs("./images", exist_ok=True)
    
    # 定义分析函数字典
    analysis_functions = {
        "difficulty": difficulty_distribution,
        "quality": quality_distribution,
        "classification": classification_distribution,
        "safety": safety_distribution,
        "rewards": reward_distribution,
        "language": language_distribution,
        "token_count": token_count_distribution
    }
    
    # 遍历每个维度进行分析和绘图
    for key, func in analysis_functions.items():
        if key in data[0]:
            distribution = func(data)
            print(f"{key}分布: {distribution}")
            
            plt.figure(figsize=(10, 6))
            plt.bar(distribution.keys(), distribution.values())
            plt.xlabel(key)
            plt.ylabel("数量")
            plt.title(f"{key}分布")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"./images/{key}_distribution.png")
            plt.close()


if __name__ == "__main__":
    data_path = sys.argv[1]
    analyze_data_distribution(data_path)