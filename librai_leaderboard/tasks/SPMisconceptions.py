from .base import BaseDirectRiskyTaskNoSeed

# paper: https://dl.acm.org/doi/10.1145/3627106.3627196
# github: https://github.com/purseclab/LLM_Security_Privacy_Advice/tree/main

class SPMisconceptions(BaseDirectRiskyTaskNoSeed):
    task_name = "sp_misconceptions"
    task_data_file = "SPMisconceptions.jsonl"
