from .base import BaseDirectRiskyTaskNoSeed

# Dataset from https://github.com/bertiev/SimpleSafetyTests
class SimpleSafetyTest(BaseDirectRiskyTaskNoSeed):
    task_name = "simple_safety_test"
    task_data_file = "SimpleSafetyTests.jsonl"
