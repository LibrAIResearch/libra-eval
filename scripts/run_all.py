import os
from argparse import ArgumentParser
from .__init__ import model_list, task_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--client", type=str, default="openai")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_samples_per_task", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rewrite_cache", action="store_true")
    args = parser.parse_args()

    if args.model == "all":
        models = model_list
    else:
        assert args.model in model_list
        models = [args.model]
    if args.task == "all":
        tasks = task_list
    else:
        assert args.task in task_list
        tasks = [args.task]

    for model in models:
        for task in tasks:
            print(f"Running {model} on {task} with debug={args.debug}")
            cmd = f"python -m librai_leaderboard.run_eval --model {model} --task {task} --client {args.client} --output_dir {args.output_dir}"
            if args.debug:
                cmd += " --debug"
            if args.rewrite_cache:
                cmd += " --rewrite_cache"
            if args.n_samples_per_task is not None:
                cmd += f" --n_samples_per_task {args.n_samples_per_task}"
            os.system(cmd)


if __name__ == "__main__":
    main()
