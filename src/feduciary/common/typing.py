from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
from torch.nn import Parameter
from torch import Tensor
import typing as t
from torch.utils.data import Subset, Dataset

ClientIds_t =  list[str]
ActorParams_t = dict[str, Parameter]
ActorParamlist_t = list[Parameter]

# dictionary of all clients' parameter sets
ClientParams_t = dict[str, ActorParams_t]
ActorDeltas_t = dict[str, Tensor]
ClientDeltas_t = dict[str, ActorDeltas_t]

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
    actor: str = '' # Who is constructing this result
    event: str = '' # What is this event for this result : train/test/val, local/central
    phase: str = '' # When was this result constructed : before aggregation/ before communication/ befor decryption
    _round: int = -1 # -1 to show that the actor has not set the round
    size: int = 0  # dataset size used to generate this result object
    metrics: dict[str, float] = field(default_factory=dict) # values that need logging
    metadata: dict[str, t.Any] = field(default_factory=dict) # values that are needed for processing/consumption

@dataclass
class ClientResult:
    params: dict[str, Parameter] = field(default_factory=dict)
    result: Result = field(default_factory=Result)


@dataclass
class ClientIns:
    # Fields set by the strategy
    params: dict[str, Parameter] = field(default_factory=dict)
    # param_keys: list[str] = field(default_factory=list)
    metadata: dict[str, t.Any] = field(default_factory=dict)
    # Reserved fields to be set by the server
    _round: int = -1
    request: RequestType = RequestType.NULL

RequestOutcomes_t = dict[str, RequestOutcome]
ClientResults_t = dict[str, ClientResult]
Results_t = dict[str, Result]

ClientIns_t = dict[str, ClientIns]

ClientDatasets_t = list[tuple[Subset, Subset]]
DatasetPair_t= tuple[Subset, Subset]

