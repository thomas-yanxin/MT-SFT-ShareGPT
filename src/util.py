import json
import os
import sys

import jsonlines
from pydantic import BaseModel
from tqdm import tqdm


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


def load_jsonl_to_list(jsonl_file_path):
    data_list = []
    with open(jsonl_file_path, "r", encoding="utf8") as file:
        for line in jsonlines.Reader(file):
            text = ""
            for i in line["conversations"]:
                text += i["value"] + " "
            line["text"] = text
            data_list.append(line)
    return data_list


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
                            if line[key] in [
                                "very easy",
                                "easy",
                                "medium",
                                "hard",
                                "very hard",
                            ]:
                                refined[key] = line[key]
                            else:
                                refined[key] = "medium"
                        elif key == "classification":
                            if line[key] in [
                                "Information seeking",
                                "Reasoning",
                                "Planning",
                                "Editing",
                                "Coding & Debugging",
                                "Math",
                                "Role playing",
                                "Data analysis",
                                "Creative writing",
                                "Advice seeking",
                                "Brainstorming",
                                "Others",
                            ]:
                                refined[key] = line[key]
                            else:
                                refined[key] = "Others"
                        elif key == "quality":
                            # [very poor/poor/average/good/excellent]
                            if line[key] in [
                                "very poor",
                                "poor",
                                "average",
                                "good",
                                "excellent",
                            ]:
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




class Difficulty(BaseModel):
    intent: str
    knowledge: str
    difficulty: str

class Classification(BaseModel):
    primary_tag: str
    other_tags: list

class Quality(BaseModel):
    explanation: str
    input_quality: str


def input_difficulty_rating(input):
    user_message = f"""
# Instruction 

You first need to identify the given user intent and then label the difficulty level of the user query based on the content of the user query.

## User Query
```
{input}
```

## Output Format
Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query.
Then, rate the difficulty level of the user query as `very easy`, `easy`, `medium`, `hard`, or `very hard`.

Now, please output the user intent and difficulty level below in a json format by filling in the placeholders in []:
```
{{
    "difficulty": "[very easy/easy/medium/hard/very hard]",
    "intent": "The user wants to [....]",
    "knowledge": "To solve this problem, the models need to know [....]",
}}
```
"""
    return user_message



def input_classification(input):
    user_message = f"""
# Instruction

Please label the task tags for the user query.

## User Query
```
{input}
```

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and select the most relevant task tag from the list below.

all_task_tags = [
    "Information seeking",  # Users ask for specific information or facts about various topics.
    "Reasoning",  # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",  # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",  # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",  # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",  # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
    "Data analysis",  # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",  # Users seek assistance with crafting stories, poems, or other creative texts. 
    "Advice seeking",  # Users ask for recommendations or guidance on various personal or professional issues.
    "Brainstorming",  # Involves generating ideas, creative thinking, or exploring possibilities. 
    "Others"  # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]

## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
```
{{
    "primary_tag": "<primary tag>",
    "other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```
"""
    return user_message



def input_quality_rating(input):
    user_message = f"""
# Instruction

You need to rate the quality of the user query based on its clarity, specificity, and coherence.

The rating scale is as follows:

- very poor: The query is unclear, vague, or incoherent. It lacks essential information and context.
- poor: The query is somewhat unclear or lacks important details. It requires significant clarification.
- average: The query is moderately clear and specific. It may require some additional information for a complete understanding.
- good: The query is clear, specific, and mostly well-formed. It provides sufficient context for understanding the user's intent.
- excellent: The query is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response.

## User Query
```
{input}
```

## Output Format
Given the user query, you first need to give an assesement, highlighting the strengths and/or weaknesses of the user query.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
```
{{   
    "input_quality": "[very poor/poor/average/good/excellent]",
    "explanation": "[...]"
}}
```
"""
    return user_message
