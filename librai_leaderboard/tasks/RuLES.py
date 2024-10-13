"""Evaluation script for LLM RuLES task

# paper: https://arxiv.org/pdf/2311.04235
# github: https://github.com/normster/llm_rules?tab=readme-ov-file

"""

import os
import json
import pandas as pd

from importlib import resources
import json
import os
import yaml

from .base import Task
from .utils.llm_rules import Message, Role, scenarios
import logging
from datetime import datetime
# Set up logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO)
logging.info(f"Pipeline started at {datetime.now()}")

with resources.files("librai_leaderboard").joinpath("tasks/utils/llm_rules/metadata.yaml").open() as f:
    metadata = yaml.safe_load(f)

class RuLES(Task):
    task_name = "llm_rules"
    task_data_file = "RuLES.jsonl"

    def _single_input(self, instance, llm_client):
        logging.info(f"Processing single input for instance: {instance['scenario_name']}")
        test_messages = Message.unserialize(instance["messages"])
        scenario_name = instance["scenario_name"]
        scenario = scenarios.SCENARIOS[scenario_name](instance["params"])

        full_prompt = scenario.prompt

        messages = [
            Message(Role.SYSTEM, ""),
            Message(Role.USER, full_prompt),
            Message(Role.ASSISTANT, scenario.initial_response),
        ]
        
        # Skip over prefilled assistant messages
        if any([m.role == Role.ASSISTANT for m in test_messages]):
            last_idx = max(
                [i for i, m in enumerate(test_messages) if m.role == Role.ASSISTANT]
            )
            messages = messages + test_messages[: last_idx + 1]
            test_messages = test_messages[last_idx + 1 :]
        # starting from the first user message and ending the second last message
        for i, m in enumerate(test_messages[1:-1]):
            messages.append(m)
            response = llm_client._call(Message.serialize(messages))
            messages.append(Message(Role.ASSISTANT, response))

        messages.append(test_messages[-1])
        logging.info(f"Completed processing single input for instance: {instance['scenario_name']}")
        print("check")
        return Message.serialize(messages)

    def _single_eval_message(self, instance):
        logging.info(f"Evaluating message for instance: {instance['scenario_name']}")
        messages = Message.unserialize(instance["input"])
        messages.append(Message(Role.ASSISTANT, instance["response"]))

        scenario = scenarios.SCENARIOS[instance["scenario_name"]](instance["params"])
        
        result = scenarios.Result(True)
        result = scenario.evaluate(messages, system_instructions=False)

        logging.info(f"Completed evaluation for instance: {instance['scenario_name']} with result: {result.passed}")
        return result.passed

    def _single_eval_postprocess(self, eval_response):
        return eval_response
    
    def run_pipeline(self, llm_client, llm_eval_client=None, rewrite_cache=False):

        # Get model responses (check if responses are already saved in a file)
        model_name = llm_client.model.split("/")[-1]
        response_fpath = os.path.join(self.response_dir, self.task_name + "_" + model_name + ".jsonl")
        batch_fpath = os.path.join(self.tmp_dir, "batch_" + self.task_name + "_" + model_name + ".jsonl")
        eval_fpath = os.path.join(self.eval_dir, self.task_name + "_" + model_name + ".jsonl")
        eval_msg_fpath = os.path.join(self.tmp_dir, "sample_eval_msg_" + self.task_name + "_" + model_name + ".jsonl")
        result_fpath = os.path.join(self.results_dir, self.task_name + "_" + model_name + ".json")

        if not rewrite_cache and os.path.exists(result_fpath):
            with open(result_fpath, "r") as f:
                result = json.load(f)
            logging.info(f"Result loaded from cache at {datetime.now()}")
            return result["score"]

        if not rewrite_cache and os.path.exists(response_fpath):
            self.data_df = pd.read_json(response_fpath, lines=True)
            logging.info(f"Responses loaded from cache at {datetime.now()}")
        else:
            inputs = self.data_df.apply(lambda instance: self._single_input(instance, llm_client), axis=1)
            responses = llm_client.multi_call(inputs)
            self.data_df["input"] = inputs
            self.data_df["response"] = responses
            self.data_df.to_json(response_fpath, orient="records", lines=True)
            logging.info(f"Responses generated and saved at {datetime.now()}")

        # check again for empty responses
        n_empty_responses = self.data_df["response"].apply(lambda x: x == "").sum()
        if n_empty_responses > 0:
            condition = self.data_df["response"] == ""
            inputs = self.data_df.loc[condition, "input"].tolist()
            responses = llm_client.multi_call(inputs)
            self.data_df.loc[condition, "response"] = responses
            self.data_df.to_json(response_fpath, orient="records", lines=True)
            logging.info(f"Empty responses filled and saved at {datetime.now()}")

        n_empty_responses = self.data_df["response"].apply(lambda x: x == "").sum()
        if n_empty_responses / len(self.data_df) > 0.1:
            logging.error(f"Too many empty responses ({n_empty_responses}/{len(self.data_df)}) at {datetime.now()}")
            raise Exception(f"Too many empty responses ({n_empty_responses}/{len(self.data_df)})")
        else:
            self.data_df = self.data_df[self.data_df["response"] != ""]
            logging.info(f"Filtered out empty responses at {datetime.now()}")

        if not rewrite_cache and os.path.exists(eval_fpath):
            self.data_df = pd.read_json(eval_fpath, lines=True)
            logging.info(f"Evaluation results loaded from cache at {datetime.now()}")
        else:
            if self.llm_eval:
                eval_messages = self.data_df.apply(self._single_eval_message, axis=1)
                eval_response = self.fake_openai_batch_request(eval_messages, batch_fpath, llm_eval_client)
                with open(eval_msg_fpath,"w") as f:
                    f.write(str(eval_messages.iloc[0]))
                logging.info(f"Evaluation messages generated and saved at {datetime.now()}")
            else:
                eval_response = self.data_df.apply(self._single_eval_message, axis=1)
                self.data_df["eval_response"] = eval_response
                self.data_df.to_json(eval_fpath, orient="records", lines=True)
                logging.info(f"Evaluation responses saved at {datetime.now()}")

        # Join the scenario_name and behavior_name columns with an underscore
        self.data_df["task"] = self.data_df.apply(
            lambda row: f"{row['scenario_name']}_{row['behavior_name']}" if row["behavior_name"] else row["scenario_name"], axis=1
        )
        self.data_df["task_nature"] = self.data_df["task"].apply(lambda x: metadata[x]["category"])
        
        harmless_score = self.data_df[self.data_df["task_nature"] == "harmless"]["eval_response"].mean()
        helpful_score = self.data_df[self.data_df["task_nature"] == "helpful"]["eval_response"].mean()

        final_score = self.data_df["eval_response"].apply(self._single_eval_postprocess).mean()

        logging.info(f"Pipeline completed at {datetime.now()} with final score: {final_score}, harmless score: {harmless_score}, helpful score: {helpful_score}")

        with open(result_fpath, "w") as f:
            json.dump({"task": self.task_name, "model": model_name, "score": final_score, "harmless_score": harmless_score, "helpful_score": helpful_score}, f, indent=2)
        return final_score