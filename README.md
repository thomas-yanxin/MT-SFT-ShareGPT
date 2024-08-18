<h1 align="center">🐋MT-SFT-ShareGPT</h1>

### Introduction

Data has always been an important part of advancing large language models forward. Based on this, we have collected dozens of high-quality open source datasets from the open source community, with a total data volume of **20 M**. 

After some cleaning actions, we have open sourced a set of high-quality datasets for fine-tuning the instructions of the Large Language Models, which are divided into three major categories, namely **English**, **Chinese**, and **Other Languages**, and **13 subcategories** according to different tasks. 

The total data size of the dataset is **5,563,444**, which can be seen in the following table:

- EN

| Classification | **Total** | Information seeking | Reasoning | Planning | Editing | Coding | Math | Role playing | Data analysis | Creative writing | Advice seeking | Brainstorming | Translation | Others |
| :----: |  :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Samples   | **3,895,845**| 571,468             | 326,777   | 76,477   | 44,163  | 926,521| 1,339,602    | 10,636        | 19,597           | 62,150          | 17,197        | 1,821      | 3,357   | 496,079 |
| tokens | **437.14**   | 558                | 371       | 819      | 474     | 476    | 344         | 518           | 550              | 808             | 622           | 571        | 155     | 398    |
| score | **0.29**   | 0.28               | 0.24      | 0.32     | 0.22    | 0.28   | 0.30        | 0.23          | 0.19             | 0.39            | 0.46          | 0.36       | 0.51    | 0.27   |
| turns | **1.21**    | 1.78               | 1.16      | 1.84     | 1.43    | 1.01   | 1.02        | 1.81          | 1.25             | 1.84            | 1.79          | 1.63       | 1.10    | 1.17   |

- CN

| Classification | **Total** | Information seeking | Reasoning | Planning | Editing | Coding | Math | Role playing | Data analysis | Creative writing | Advice seeking | Brainstorming | Translation | Others |
| :----: |  :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Samples   | **1,635,223** | 558,187             | 237,906   | 45,327   | 94,058  | 87,689  | 109,694     | 37,695        | 16,578           | 175,587         | 30,656        | 4,620      | 0       | 237,226 |
| tokens | **272.79**   | 290.12             | 244.53    | 356.71   | 194.36  | 404     | 273.10      | 287           | 313              | 276             | 286           | 316        | 0       | 216.16  |
| score | **0.35**     | 0.34               | 0.36      | 0.25     | 0.42    | 0.77    | 0.24        | -0.01         | 0.12             | 0.36            | 0.380         | -0.03      | 0       | 0.33    |
| turns | **1.20**     | 1.09               | 1.05      | 1.14     | 1.08    | 1.17    | 1.11        | 1.10          | 1.14             | 1.91            | 1.09          | 2.87       | 0       | 1.17    |

- Others

| Classification | **Total** | Information seeking | Reasoning | Planning | Editing | Coding | Math | Role playing | Data analysis | Creative writing | Advice seeking | Brainstorming | Translation | Others |
| :----: |  :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Samples | **32,376**   | 10,504             | 2,310     | 663      | 895     | 3,081   | 2,050       | 147           | 118              | 1,041           | 483           | 38         | 0       | 11,046  |
| tokens | **452.52**   | 443.44             | 502.11    | 825.08   | 577.28  | 545.38  | 450.24      | 858.58        | 793.2            | 752.52          | 608.77        | 773.05     | 0       | 347.57  |
| score | **0.12**     | -0.08              | -0.1      | 0.2      | 0.16    | 0.17    | 0.04        | 0.11          | 0.11             | 0.24            | -0.1          | -0.26      | 0       | 0.36    |
| turns | **1.30**     | 1.31               | 1.35      | 1.62     | 1.48    | 1.25    | 1.08        | 1.47          | 1.47             | 1.66            | 1.57          | 1.92       | 0       | 1.23    |

### Data Processing

We used the [internlm/internlm2-reward](https://huggingface.co/internlm/internlm2-20b-reward) series and [sfairXC/FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) to score each Sample, using the [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) series to categorise the data for tasks, quality scoring and other operations, use [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) to determine if Sample meets security standards.

Each Sample conforms to the **ShareGPT** format, which is the prevailing specification in the community for training large language models:

```
{"id": "Infinity-Instruct_3_+560470", "conversations": [{"from": "human", "value": "Explain the specific steps you take when dealing with a coworker who is resistant to changing their work habits or processes. Include examples of communication techniques, conflict resolution strategies, and any other tools or resources you find helpful in these situations."}, {"from": "gpt", "value": "I don't have direct experience in dealing with people, but here are some steps you can take when dealing with a coworker who is resistant to changing their work habits or processes:\n\n1. identify the reasons for the resistance: the first step is to understand why your coworker is resistant to change. they may feel like their processes are the most efficient or they may be resistant to change because they fear they won't be able to adapt quickly.\n\n2. active listening: listen actively to your coworker and try to understand their point of view.\n\n3. communicate the need for change: explain why the change is necessary and the benefits it will bring to the team, department, or company as a whole.\n\n4. provide adequate training: offer enough training to ensure that your coworker understands the new process or procedure and feels confident in adapting to the change.\n\n5. identify support systems: identify support systems and resources available for your coworker that will help them learn and adapt to the new processes.\n\n6. conflict management: if there is a conflict, address it calmly and respectfully, trying to understand their concerns and offer a compromise that works for both parties.\n\n7. encouragement: encourage your coworker to try out the new process, giving them a chance to test the change on a small scale before implementing it fully.\n\n8. follow-up and feedback: conduct follow-up sessions to ensure that your coworker is adapting well to the change and is satisfied with the new process or procedures.\n\nby following these steps, you can effectively handle a coworker who is resistant to changing their work habits and ultimately foster a more productive, efficient work environment."}, {"from": "human", "value": "Can you please provide some examples of communication techniques that might work well in situations like this?"}, {"from": "gpt", "value": "Certainly! Here are some specific communication techniques that could be helpful when dealing with a coworker who is resistant to changing their work habits or processes:\n\n1. Be empathetic: Try to understand and acknowledge your coworker's point of view. Let them know that you appreciate their current processes and that the objective of the change isn't to minimize or detract from their work. 2. Provide clear communication: Explain the reasons for the change in a clear and concise way. Avoid jargons, and provide examples of the benefits of the changes so that your coworker can understand the purpose behind the changes. 3. Active listening: Listen to your coworker actively so that you understand their concerns and ideas. Hear them out, and ask clarifying questions if necessary. 4. Encourage feedback: Encourage your coworker to provide feedback on the new process. Ask them how the new system can be made better, and if there are any areas they might want to tweak to make it work for them. 5. Be patient: Changing work habits can take time. Be patient, and give your coworker time to adjust to the new system. 6. Provide incentives: Consider providing incentives such as bonuses or promotions for adapting to the changes. By utilizing these effective communication techniques, you can effectively manage a reluctant coworker and help them understand the importance and benefits of the change."}, {"from": "human", "value": "Can you give me some more examples of conflict resolution strategies that could be helpful in these situations?"}, {"from": "gpt", "value": "Sure! Here are some conflict resolution strategies that could help in situations where you're dealing with a coworker who is resistant to changing their work habits or processes:\n\n1. Address the issue directly: If there is a conflict, address it head-on but in a calm and respectful way. Try to understand their perspective and find a compromise that works well for both parties. 2. Define the issue objectively: Make sure to accurately define the issue and make sure that both parties understand the challenge at hand. 3. Brainstorm solutions together: Sit down together with your coworker and brainstorm solutions to the issue. Create a supportive and collaborative environment where suggestions and feedback are encouraged. 4. Focus on common goals: Communicate the common goals clearly and demonstrate how the change can help the team or company achieve them. 5. Be patient: Understand that change can be difficult for some people and adaptability can take time. Be patient, and try to expect doubts and uncertainty and be prepared to answer them. 6. Get a third-party mediator: If you're unable to solve the issue on your own, consider engaging a neutral 3rd party mediator to help solve the issue. 7. Create action plans: Plan out clear, actionable steps for yourself and your coworker to take to ensure that you're both working towards a common goal. By adopting these conflict resolution strategies, you can ensure that your coworker is more open to collaborating and striving for mutually beneficial resolutions to implement changes to their work habits or processes."}], "source": "Infinity-Instruct_3", "token_count": 1044, "rewards": 4.172325134277344e-07, "difficulty": "medium", "classification": "Advice seeking", "quality": "excellent", "language": "EN", "safety": "safe"}
```


### Thanks

- models
1. [internlm/internlm2-reward](https://huggingface.co/internlm/internlm2-20b-reward)
2. [sfairXC/FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1)
3. [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
4. [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

- datasets
1. [argilla/magpie-ultra-v0.1](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1)
2. [BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)
3. [Replete-AI/code_bagel_hermes-2.5](https://huggingface.co/datasets/Replete-AI/code_bagel_hermes-2.5)
4. [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)
5. [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
6. [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
7. [arcee-ai/The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome)
8. ……
   
- Papers
1. https://arxiv.org/pdf/2304.12244
2. https://arxiv.org/pdf/2406.08464
3. https://ai.meta.com/research/publications/the-llama-3-herd-of-models/
