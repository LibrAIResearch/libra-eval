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
    PersonalInfoLeak_five_shot,
    PersonalInfoLeak_five_shot_non_domain,
    PersonalInfoLeak_one_shot,
    PersonalInfoLeak_one_shot_non_domain,
    PersonalInfoLeak_two_shot,
    PersonalInfoLeak_two_shot_non_domain,
    PersonalInfoLeak_zero_shot_a,
    PersonalInfoLeak_zero_shot_b,
    PersonalInfoLeak_zero_shot_c,
    PersonalInfoLeak_zero_shot_d,
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
    SycophancyEval,
    SycophancyEvalAnswer,
    SycophancyEvalAreYouSure,
    SycophancyEvalFeedback,
    SycophancyEvalMimicry,
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
from .SafeText import SafeText
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
    "personalInfoLeak-five_shot_non_domain": PersonalInfoLeak_five_shot_non_domain,
    "personalInfoLeak-five_shot": PersonalInfoLeak_five_shot,
    "personalInfoLeak-one_shot_non_domain": PersonalInfoLeak_one_shot_non_domain,
    "personalInfoLeak-one_shot": PersonalInfoLeak_one_shot,
    "personalInfoLeak-two_shot_non_domain": PersonalInfoLeak_two_shot_non_domain,
    "personalInfoLeak-two_shot": PersonalInfoLeak_two_shot,
    "personalInfoLeak-zero_shot-a": PersonalInfoLeak_zero_shot_a,
    "personalInfoLeak-zero_shot-b": PersonalInfoLeak_zero_shot_b,
    "personalInfoLeak-zero_shot-c": PersonalInfoLeak_zero_shot_c,
    "personalInfoLeak-zero_shot-d": PersonalInfoLeak_zero_shot_d,
    "prompt_extraction_robustness": PromptExtractionRobustness,
    "prompt_hijacking_robustness": PromptHijackingRobustness,
    "qharm": QHarm,
    "moral_choice": MoralChoice,
    "hypothesis_stereotypes": HypothesisStereotypes,
    "latent_jailbreak": LatentJailbreak,
    "malicious_instruct": MaliciousInstruct,
    "strong_reject": StrongREJECT,
    "sycophancy_eval": SycophancyEval,
    "sycophancy_eval_answer": SycophancyEvalAnswer,
    "sycophancy_eval_are_you_sure": SycophancyEvalAreYouSure,
    "sycophancy_eval_feedback": SycophancyEvalFeedback,
    "sycophancy_eval_mimicry": SycophancyEvalMimicry,
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
    "safe_text": SafeText,
    "hex_phi": HExPHI,
    "TruthfulQA_mc1": TruthfulQA_mc1,
    "TruthfulQA_mc2": TruthfulQA_mc2,
    "TruthfulQA_gen": TruthfulQA_gen,
    "XSTest": XSTest,
    "Toxigen": Toxigen,
    "aart": Aart,
    "safety_prompt": SafetyPrompt,
    "simple_safety_test": SimpleSafetyTest,
    "sp_misconceptions": SPMisconceptions,
    "anthropic_harmless_base": AnthropicHarmlessBase,
    "dices350": DICES350,
    "bbq": BBQ,
}

TASK_GROUPS = {}
