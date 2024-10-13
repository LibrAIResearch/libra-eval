from .base import BaseDirectRiskyTask

# paper: https://arxiv.org/abs/2309.07875
# github: https://github.com/vinid/instruction-llms-safety-eval

class QHarm(BaseDirectRiskyTask):
    task_name = "qharm"
    task_data_file = "QHarm.jsonl"
