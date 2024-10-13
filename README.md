# librai-leaderboard-data

## Introduction

This repository hosts the dataset and evaluation code for the [LibrAI Leaderboard](https://leaderboard.librai.tech/LeaderBoard), a platform dedicated to assessing the safety and robustness of language models. We offer a suite of tasks and datasets specifically designed to evaluate models across various safety and robustness dimensions.

**Important Note:** This is an ongoing research project. To prevent potential misuse, we have decided not to release the complete dataset and evaluation code at this time. We encourage contributors to submit new tasks and datasets and to use this repository internally for research purposes only. Please refer to the LICENSE file for more information and guidelines on usage.

## Guideline: How to Add a New Task

### Step 1: Create a Dataset

Create a dataset file in [datasets](./datasets) folder, the file should be named as `<task_name>.jsonl`.
Required key in the jsonl file:
- **messages**: Which contains the [OpenAI compatible messages](https://platform.openai.com/docs/guides/chat-completions/getting-started). Each message should have a `role` and `content` field. The `role` can be `system`, `user`, or  `assistant`. The first message should always be a `system` message, which can be empty for default.
- Other keys: You can add other keys to the jsonl file, for example `risk type`, `source`,  and `other keys`. Try keep as much information as possible in the jsonl file.

Please refer to the notebook [tutorial](format_data.ipynb) for more information on how to format the data.

### Step 2: Create a Task Class

Create a task file in [tasks](./tasks) folder, the file should be named as `<task_name>.py`.
Each task is a python class, which should inherit from **Task** in [tasks/base.py](./tasks/base.py). We have implemented the **BaseDirectRiskyTask** for you. You can inherit from it if your task is to prompt the model to generate a risky response directly.

Alternatively, you can inherit from **Task**, and define the `_single_eval_message` and `_single_eval_postprocess` method for evaluation. The input field `instance` is the json object in the dataset file. 

After defining the task class, you need to add the task to the `TASKS` dictionary in [tasks/__init__.py](./tasks/__init__.py).

> **Note:**
> Please try to follow the evaluation method in the original paper. For example, some dataset use string matching, some use LLM-as-a-judge. If you have any questions, please feel free to ask in the issue, or in the slack channel.

### Step 3: Test

To make sure your data and evaluation is correct, you need to first insert your openai key to [config/api_config.json](./config/api_config.json).

**Environment:** Our code requires openai>=1.0.0, and pandas. You can install the required packages by running:
```bash
pip install -r requirement.txt
```


After that, run the following command:
```bash
python -m librai_leaderboard.run_eval --client openai --model gpt-4o-mini --task <your_task_name> --debug --rewrite_cache
# Please note your_task_name is define in tasks/__init__.py and may different from the file name.
```
Usually, it will finish within 1 minute, and cost less than 0.1 USD. 

If you worry about the correctness of the framework and environment, you can run the following command to test DoNotAnswer task:
```bash
python -m librai_leaderboard.run_eval --client openai --model gpt-4o-mini --task do_not_answer --debug --rewrite_cache
```

Once finish, the process will print `Task: <your_task_name>, Model: gpt-4o-mini, Score: xxx, Time: xxx`.
The results will be output to `outputs` folder. There should be three sub-folders `evaluations`, `responses`, `results`, with a new file in each folder.

> **Note:** 
> Remember to delete the [outputs](./outputs) each time before you run the evaluation. Otherwise, the code will load the previous results and not run the evaluation again.

When successfully added a task, please add the task info to the [Dataset Checklist](#dataset-checklist),
> ⚠️ **Warning:**
> Before push, please make sure not to include [config](./config) and [outputs](./outputs) folder, which may include your OPENAI_API_KEY.



## Dataset Checklist

- ExampleTaskA
  - [x] Data
  - [ ] Evaluation
- ExampleTassB
  - [x] Data
  - [x] Evaluation
- AdvBench
  - [x] Data
  - [x] Evaluation
- DoNotAnswer
  - [x] Data
  - [x] Evaluation
- I_CoNa
  - [x] Data
  - [x] Evaluation
- MITRE
  - [x] Data
  - [x] Evaluation
- PromptInjection
  - [x] Data
  - [x] Evaluation
- PromptExtractionRobustness
  - [x] Data
  - [x] Evaluation
- PromptHijackingRobustness
  - [x] Data
  - [x] Evaluation
- QHarm
  - [x] Data
  - [x] Evaluation
- HypothesisStereotypes
  - [x] Data
  - [x] Evaluation
- LatentJailbreak
  - [x] Data
  - [x] Evaluation
- MaliciousInstruct
  - [x] Data
  - [x] Evaluation
- StrongREJECT
  - [x] Data
  - [x] Evaluation
- SycophancyEval
  - [x] Data
  - [x] Evaluation
- TDCRedTeaming
  - [x] Data
  - [x] Evaluation
- DoAnythingNowForbiddenQuestions
  - [x] Data
  - [x] Evaluation
- GandalfIgnoreInstructions
  - [x] Data
  - [x] Evaluation
- TruthfulQA
  - [x] Data
  - [x] Evaluation
- XSTest
  - [x] Data
  - [x] Evaluation
- Toxigen
  - [x] Data
  - [x] Evaluation
- AART
  - [x] Data
  - [x] Evaluation
- AnthropicHarmlessBase
  - [x] Data
  - [x] Evaluation
- ConfAIde
  - [x] Data
  - [x] Evaluation
- ControversialInstructions
  - [x] Data
  - [x] Evaluation
- CyberattackAssistance
  - [x] Data
  - [x] Evaluation
- SafeText
  - [x] Data
  - [x] Evaluation