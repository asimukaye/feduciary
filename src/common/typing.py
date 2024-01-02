from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum, auto
from torch.nn import Parameter
import typing as t

ClientIds_t =  list[str]
ClientParams_t = dict[str, OrderedDict[str, Parameter]]
ActorParams_t = OrderedDict[str, Parameter]

class RequestType(Enum):
    TRAIN = auto()
    TRAIN_VAL = auto()
    EVAL = auto()
    RESET = auto()
    NULL = auto()

class RequestOutcome(Enum):
    PENDING = auto()
    COMPLETE = auto()
    FAILED = auto()

@dataclass
class Result:
    actor: str = ''# this is just for debugging for now
    epoch: int = -1 # epoch -1 reserved for evaluation request
    _round: int = 0 # this is just for debugging for now
    size: int = 0  # dataset size used to generate this result object
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, t.Any] = field(default_factory=dict)

@dataclass
class ClientResult1:
    params: OrderedDict[str, Parameter] = field(default_factory=OrderedDict)
    result: Result = field(default=Result())


@dataclass
class ClientIns:
    params: OrderedDict[str, Parameter] = field(default_factory=OrderedDict)
    metadata: OrderedDict[str, t.Any] = field(default_factory=OrderedDict)
    # Reserved to be set by the server
    _round: int = -1
    request: RequestType = RequestType.NULL


ClientResults_t = dict[str, ClientResult1]
EvalResults_t = dict[str, Result]

ClientIns_t = dict[str, ClientIns]

# ClientParams_t = dict[str, OrderedDict[str, torch.nn.Parameter]]
