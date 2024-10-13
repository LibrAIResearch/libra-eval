from .base import BaseDirectRiskyTask

# github: https://github.com/google-research-datasets/aart-ai-safety-dataset
# paper: https://aclanthology.org/2023.emnlp-industry.37/

class Aart(BaseDirectRiskyTask):
    task_name = "aart"
    task_data_file = "Aart.jsonl"
