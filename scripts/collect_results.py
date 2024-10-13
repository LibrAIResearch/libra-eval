import os
import json
import datetime
from tabulate import tabulate
from argparse import ArgumentParser
from .__init__ import model_list, task_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

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

    summary = {}  # model -> task -> score
    headers = ["Model"] + [n[-8:] for n in list(tasks)]
    all_results = []
    for model in models:
        scores = {"Arena Elo": 0}
        row = [model]


        for task in list(tasks):
            if args.debug:
                result_fpath = os.path.join(script_dir, f"../librai_leaderboard/outputs/results/debug_{task}_{model}.json")
            else:
                result_fpath = os.path.join(script_dir, f"../librai_leaderboard/outputs/results/{task}_{model}.json")
            if os.path.exists(result_fpath):
                with open(result_fpath, "r") as f:
                    result = json.load(f)
                    row.append(f"{result['score']*100:.0f}")
                    scores[task] = result["score"]
            else:
                row.append("N/A")
                print("Missing result file: ", result_fpath)
                scores[task] = "N/A"

        all_results.append(row)
        summary[model] = scores
    print(tabulate(all_results, headers, tablefmt="pipe", stralign="left"))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(script_dir, f"../librai_leaderboard/outputs/summary_{timestamp}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
