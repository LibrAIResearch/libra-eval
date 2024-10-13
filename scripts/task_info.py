from librai_leaderboard.tasks import TASKS

for task in TASKS.keys():
    instance = TASKS[task](debug=False)
    print(instance.task_name, len(instance.data_df))
