from .base import BaseDirectRiskyTaskNoSeed

class AttackCiphering(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_ciphering"
    task_data_file = "librai_adv_ciphering.jsonl"


class AttackDeepInception(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_deep_inception"
    task_data_file = "librai_adv_deep_inception.jsonl"


class AttackDoAnythingNow(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_do_anything_now"
    task_data_file = "librai_adv_do_anything_now.jsonl"


class AttackEffectToCause(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_effect_to_cause"
    task_data_file = "librai_adv_effect_to_cause.jsonl"


class AttackFewShots(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_few_shots"
    task_data_file = "librai_adv_few_shots.jsonl"


class AttackMultilingual(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_multilingual"
    task_data_file = "librai_adv_multilingual.jsonl"


class AttackOneSidedStatement(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_one_sided_statement"
    task_data_file = "librai_adv_one_sided_statement.jsonl"


class AttackPersonaModulation(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_persona_modulation"
    task_data_file = "librai_adv_persona_modulation.jsonl"


class AttackPromptInjection(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_prompt_injection"
    task_data_file = "librai_adv_prompt_injection.jsonl"


class AttackRefusalSuppression(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_refusal_suppression"
    task_data_file = "librai_adv_refusal_suppression.jsonl"


class AttackTenseChange(BaseDirectRiskyTaskNoSeed):
    task_name = "librai_adv_tense_change"
    task_data_file = "librai_adv_tense_change.jsonl"
