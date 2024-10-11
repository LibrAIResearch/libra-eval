from .Advbench import AdvBench
from .BeaverTails import BeaverTailsBad, BeaverTailsGood
from .DoNotAnswer import DoNotAnswer, DoNotAnswerFP, DoNotAnswerFN
from .CoNa import CoNa
from .PromptInjection import PromptInjection
from .DecodingTrust import (
    DecodingTrustStereotype,
    DecodingTrustPrivacy,
    DecodingTrustMachineEthics
)
from .DialogueSafety import DialogueSafety
from .ConfAIde import ConfAIde
from .ControversialInstructions import ControversialInstructions
from .CyberattackAssistance import CyberattackAssistance
from .PhysicalSafetyInstructions import (
    PhysicalSafetyInstructionsSafe,
    PhysicalSafetyInstructionsUnsafe,
)
from .MaliciousInstructions import MaliciousInstructions
from .PersonalInfoLeak import (
    PersonalInfoLeak_context,
    PersonalInfoLeak_zero_shot,
    PersonalInfoLeak_few_shot,
)
from .MoralChoice import MoralChoice
from .PromptExtractionRobustness import PromptExtractionRobustness
from .PromptHijackingRobustness import PromptHijackingRobustness
from .QHarm import QHarm
from .HypothesisStereotypes import HypothesisStereotypes
from .LatentJailbreak import LatentJailbreak
from .MaliciousInstruct import MaliciousInstruct
from .StrongREJECT import StrongREJECT
from .SycophancyEval import (
    # SycophancyEval,
    SycophancyEvalAnswer,
    SycophancyEvalAreYouSure,
    SycophancyEvalFeedback,
    # SycophancyEvalMimicry,
)
from .TDCRedTeaming import TDCRedTeaming
from .HackAPrompt import HackAPromptClean, HackAPromptAll
from .HarmBench import HarmBench
from .HarmfulQ import HarmfulQ
from .DoAnythingNow import (
    DoAnythingNowJailbreak,
    DoAnythingNowRegular
)
from .GandalfIgnoreInstructions import GandalfIgnoreInstructions
from .RedEval import (
    RedEvalHarmfulQA,
    RedEvalDangerousQA,
)
from .HExPHI import HExPHI
from .TruthfulQA import TruthfulQA_mc1, TruthfulQA_mc2, TruthfulQA_gen
from .XSTest import XSTest
from .Toxigen import Toxigen
from .Aart import Aart
from .SimpleSafetyTest import SimpleSafetyTest
from .SPMisconceptions import SPMisconceptions
from .SafetyPrompt import SafetyPrompt
from .AnthropicHarmlessBase import AnthropicHarmlessBase
from .DICES350 import DICES350
from .BBQ import BBQ
from .AnthropicRedTeam import AnthropicRedTeam
from .BAD import BAD
from .LibrAI_Adversarial import (
    AttackCiphering,
    AttackDeepInception,
    AttackDoAnythingNow,
    AttackEffectToCause,
    AttackFewShots,
    AttackMultilingual,
    AttackOneSidedStatement,
    AttackPersonaModulation,
    AttackPromptInjection,
    AttackRefusalSuppression,
    AttackTenseChange
)


TASKS = {
    "do_not_answer": DoNotAnswer,
    "do_not_answer_fp": DoNotAnswerFP,
    "do_not_answer_fn": DoNotAnswerFN,
    "advbench": AdvBench,
    "beaver_tails_bad": BeaverTailsBad,
    "beaver_tails_good": BeaverTailsGood,
    "cona": CoNa,
    "prompt_injection": PromptInjection,
    "decoding_trust_stereotype": DecodingTrustStereotype,
    "decoding_trust_privacy": DecodingTrustPrivacy,
    "decoding_trust_machine_ethics": DecodingTrustMachineEthics,
    "dialogue_safety": DialogueSafety,
    "confaide": ConfAIde,
    "controversialinstructions": ControversialInstructions,
    "cyberattackassistance": CyberattackAssistance,
    "physical_safety_instructions_safe": PhysicalSafetyInstructionsSafe,
    "physical_safety_instructions_unsafe": PhysicalSafetyInstructionsUnsafe,
    "malicious_instructions": MaliciousInstructions,
    "personalinfoleak_context": PersonalInfoLeak_context,
    "personalinfoleak_zero_shot": PersonalInfoLeak_zero_shot,
    "personalinfoleak_few_shot": PersonalInfoLeak_few_shot,
    "prompt_extraction_robustness": PromptExtractionRobustness,
    "prompt_hijacking_robustness": PromptHijackingRobustness,
    "qharm": QHarm,
    "moral_choice": MoralChoice,
    "hypothesis_stereotypes": HypothesisStereotypes,
    "latent_jailbreak": LatentJailbreak,
    "malicious_instruct": MaliciousInstruct,
    # "strong_reject": StrongREJECT, # sourced from other datasets
    "sycophancy_eval": SycophancyEval,
    "sycophancy_eval_answer": SycophancyEvalAnswer,
    "sycophancy_eval_are_you_sure": SycophancyEvalAreYouSure,
    "sycophancy_eval_feedback": SycophancyEvalFeedback,
    # "sycophancy_eval_mimicry": SycophancyEvalMimicry, # not ready for now
    "tdc_red_teaming": TDCRedTeaming,
    "hackaprompt_clean": HackAPromptClean,
    "hackaprompt_all": HackAPromptAll,
    "harmbench": HarmBench,
    "harmfulq": HarmfulQ,
    "dan_jailbreak": DoAnythingNowJailbreak,
    "dan_regular": DoAnythingNowRegular,
    "gandalf_ignore_instructions": GandalfIgnoreInstructions,
    "redeval_harmfulqa": RedEvalHarmfulQA,
    "redeval_dangerousqa": RedEvalDangerousQA,
    "hex_phi": HExPHI,
    "truthfulqa_mc1": TruthfulQA_mc1,
    # "truthfulqa_mc2": TruthfulQA_mc2, # evaluation need revise
    "truthfulqa_gen": TruthfulQA_gen,
    "xstest": XSTest,
    "toxigen": Toxigen,
    "aart": Aart,
    # "safety_prompt": SafetyPrompt, # chinese
    "simple_safety_test": SimpleSafetyTest,
    "sp_misconceptions": SPMisconceptions,
    "anthropic_harmless_base": AnthropicHarmlessBase,
    "dices350": DICES350,
    "bbq": BBQ,
    "anthropic_redteam": AnthropicRedTeam,
    "bad": BAD,
    "librai_adv_ciphering": AttackCiphering,
    "librai_adv_deep_inception": AttackDeepInception,
    "librai_adv_do_anything_now": AttackDoAnythingNow,
    "librai_adv_effect_to_cause": AttackEffectToCause,
    "librai_adv_few_shots": AttackFewShots,
    "librai_adv_multilingual": AttackMultilingual,
    "librai_adv_one_sided_statement": AttackOneSidedStatement,
    "librai_adv_persona_modulation": AttackPersonaModulation,
    "librai_adv_prompt_injection": AttackPromptInjection,
    "librai_adv_refusal_suppression": AttackRefusalSuppression,
    "librai_adv_tense_change": AttackTenseChange,
}

TASK_GROUPS = {}
