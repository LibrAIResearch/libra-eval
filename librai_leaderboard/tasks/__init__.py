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
# from .QHarm import QHarm  # evaluation need double check, many questions are not risky
from .HypothesisStereotypes import HypothesisStereotypes
from .LatentJailbreak import LatentJailbreak
from .MaliciousInstruct import MaliciousInstruct
# from .StrongREJECT import StrongREJECT  # sourced from other datasets
from .SycophancyEval import (
    SycophancyEvalAnswer,
    SycophancyEvalAreYouSure,
    SycophancyEvalFeedback,
    # SycophancyEvalMimicry,
)
from .TDCRedTeaming import TDCRedTeaming
from .HackAPrompt import HackAPrompt  # data need double check
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
from .RuLES import RuLES
from .HExPHI import HExPHI
from .TruthfulQA import (
    TruthfulQA_mc1,
    # TruthfulQA_mc2,  # evaluation need revise
    TruthfulQA_gen
)
from .XSTest import XSTest
from .Toxigen import Toxigen
from .Aart import Aart
from .SimpleSafetyTest import SimpleSafetyTest
from .SPMisconceptions import SPMisconceptions
# from .SafetyPrompt import SafetyPrompt  # Chinese
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

TASKS = {}
for cls_name, cls_obj in list(locals().items()):
    if isinstance(cls_obj, type) and hasattr(cls_obj, 'task_name'):
        TASKS[cls_obj.task_name] = cls_obj

TASK_GROUP_NAMES = ["direct_risky", "cybersecure", "instruction_hierarchy", "adversarial", "over_sensitive"]
TASK_GROUPS = {}
