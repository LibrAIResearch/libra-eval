import os
import json
import time
from argparse import ArgumentParser

from librai_leaderboard.tasks import TASKS
from librai_leaderboard.llmclient import OpenAI_Client, Next_Client, Local_Client


def eval():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--client", type=str, default="openai")
    parser.add_argument("--task", type=str, default="do_not_answer", choices=TASKS.keys())
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_samples_per_task", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rewrite_cache", action="store_true")
    args = parser.parse_args()

    # Init
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config/api_config.json"), "r") as f:
        api_config = json.load(f)

    args.output_dir = os.path.join(current_dir, args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.client == "openai":
        llm_client = OpenAI_Client(
            model=args.model,
            api_config=api_config,
        )
    elif args.client == "local":
        llm_client = Local_Client(
            model=args.model,
            api_config=api_config,
        )
    elif args.client == "next":
        llm_client = Next_Client(
            model=args.model,
            api_config=api_config,
        )
    else:
        raise ValueError("Client not support.")

    llm_eval_client = OpenAI_Client(
        model="gpt-4o-mini",
        api_config=api_config,
    )

    print(args)

    task = TASKS[args.task](debug=args.debug, output_dir=args.output_dir, n_samples_per_task=args.n_samples_per_task)

    st = time.time()
    score = task.run_pipeline(llm_client=llm_client, llm_eval_client=llm_eval_client, rewrite_cache=args.rewrite_cache)
    et = time.time()
    print(f"Task: {args.task}, Model: {args.model}, Score: {score}, Time: {et - st}")


if __name__ == "__main__":
    eval()
