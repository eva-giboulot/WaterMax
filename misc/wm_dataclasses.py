from __future__ import annotations
import dacite
import ast

from typing import Union

import re
from dataclasses import dataclass, field, replace,InitVar
from typing import Any, Dict, List, Optional, Set, Tuple,Union
from os import path


def opt_int_or_float(n:str) -> Union[float,int,None]:
    if n.isdigit():
        #print("Is int !") 
        return(int(n))
    elif n.replace('.','',1).isdigit() and n.count('.') < 2:
        #print("Is float !") 
        return(float(n))
    elif n is None:
        return None
    else:
        raise TypeError

def int_or_float(n:str) -> Union[float,int]:
    if n.isdigit():
        #print("Is int !") 
        return(int(n))
    elif n.replace('.','',1).isdigit() and n.count('.') < 2:
        #print("Is float !") 
        return(float(n))
    else:
        raise TypeError

## DATACLASS for MMW
def str_or_none(val):
    if val is not None:
        return str(val)
    else:
        return "-"


def parse_string_val(val):
    if val == "-":
        return None
    return val


def dict_to_str(val):
    return ast.literal_eval(val)


@dataclass
class WmConfig:
    seed : int
    param1: Union[float, int]
    param2 : Union[float, int]
    bench : str
    ngram : int
    temperature :float
    
    gen_len :int

    wm : str
    mode: Optional[str] = None
    beam_search: Optional[bool] = False
    beam_chunk_size: Optional[int] = 0

    def __post_init__(self):
        self.mode = self.wm



"""
For compatibility with MarkMyWords benchmarks

"""

@dataclass(frozen=True)
class DATA_VerifierSpec:
    """
    A class representing the verifier specification.

    Attributes:
        verifier (str): The verifier to use. Defaults to 'Theoretical'.
        empirical_method (str): The empirical method to use. Defaults to 'regular'.
        log (Optional[bool]): Whether to use a log score
        gamma (Optional[float]): The gamma value to use for edit distance. Defaults to 0.
    """

    verifier: str = "Theoretical"
    empirical_method: str = "regular"
    log: Optional[bool] = None
    gamma: Optional[float] = 0

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> DATA_VerifierSpec:
        return dacite.from_dict(DATA_VerifierSpec, d)

    @staticmethod
    def from_str(s: str) -> DATA_VerifierSpec:
        return DATA_VerifierSpec.from_dict(dict_to_str(s))

@dataclass(frozen=True)
class DATA_WatermarkSpec:
    """
    Specifies how to perform the watermarking
    """

    # Random number generator type
    rng: str = "Internal"

    # Random number generator parameters
    hash_len: Optional[int] = None
    min_hash: Optional[bool] = None
    secret_key: int = field(default=0, hash=False)
    key_len: Optional[int] = None

    # Generator settings
    generator: Optional[str] = None
    tokenizer: Optional[str] = field(default=None, hash=False)
    temp: float = 1.0
    delta: Optional[float] = None
    gamma: Optional[float] = None
    skip_prob: Optional[float] = 0

    # Verifier settings
    pvalue: float = 0.01
    verifiers: List[DATA_VerifierSpec] = field(default_factory=list, hash=False)

    # Randomization settings
    randomize: bool = False
    offset: bool = False

    def to_dict(self, omit_key=False, omit_verifiers=False) -> Dict:
        d = self.__dict__
        if omit_key:
            d = {k: v for k, v in d.items() if k != "secret_key"}
        if omit_verifiers:
            d = {k: v for k, v in d.items() if k != "verifiers"}
        else:
            d = {
                k: v if k != "verifiers" else [i.__dict__ for i in v]
                for k, v in d.items()
            }
        return d

    def __str__(self) -> str:
        return str(self.to_dict(omit_key=True, omit_verifiers=False))

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str:Any]) -> DATA_WatermarkSpec:
        return dacite.from_dict(DATA_WatermarkSpec, d)

    @staticmethod
    def from_str(s: str) -> DATA_WatermarkSpec:
        return DATA_WatermarkSpec.from_dict(dict_to_str(s))

    def sep_verifiers(self) -> List[DATA_WatermarkSpec]:
        rtn = []
        for v in self.verifiers:
            rtn.append(replace(self, verifiers=[v]))
        return rtn

@dataclass(frozen=True)
class DATA_SentenceWatermarkSpec:
    """
    Specifies how to perform the watermarking
    """

    # Random number generator type
    rng: str = "Numpy"

    # Generator settings
    generator: Optional[str] = None
    tokenizer: Optional[str] = field(default=None, hash=False)
    temp: float = 1.0
    N: Optional[int] = None
    n: Optional[int] = None

    def to_dict(self) -> Dict:
        d = self.__dict__

        d = {
            k: v if k != "verifiers" else [i.__dict__ for i in v]
            for k, v in d.items()
            }
        return d
    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str:Any]) -> DATA_SentenceWatermarkSpec:
        return dacite.from_dict(DATA_SentenceWatermarkSpec, d)

    @staticmethod
    def from_str(s: str) -> DATA_SentenceWatermarkSpec:
        return DATA_SentenceWatermarkSpec.from_dict(dict_to_str(s))


@dataclass(frozen=True)
class DATA_Generation:
    """Defines the content of a generation"""

    watermark: Optional[DATA_SentenceWatermarkSpec] = None
    key: Optional[int] = None
    attack: Optional[str] = None
    id: int = 0
    prompt: str = ""
    response: str = ""
    rating: Optional[float] = None
    pvalue: Optional[float] = None
    efficiency: Optional[float] = None
    token_count: int = 0
    entropy: Optional[float] = None
    spike_entropy: Optional[float] = None
    temp: Optional[float] = None

    @staticmethod
    def keys() -> List[str]:
        return [
            "watermark",
            "key",
            "id",
            "attack",
            "prompt",
            "response",
            "rating",
            "pvalue",
            "efficiency",
            "token_count",
            "entropy",
            "spike_entropy",
            "temp",
        ]

    def __str__(self) -> str:
        cpy_dict = {k: v for k, v in self.__dict__.items()}
        if "response" in cpy_dict and cpy_dict["response"] is not None:
            cpy_dict["response"] = (
                cpy_dict["response"]
                .replace("\000", "")
                .replace("\r", "___LINE___")
                .replace("\t", "___TAB___")
                .replace("\n", "___LINE___")
            )
        if "prompt" in cpy_dict and cpy_dict["prompt"] is not None:
            cpy_dict["prompt"] = (
                cpy_dict["prompt"]
                .replace("\000", "")
                .replace("\r", "___LINE___")
                .replace("\t", "___TAB___")
                .replace("\n", "___LINE___")
            )
        return "\t".join(
            [
                str_or_none(cpy_dict[v] if v in cpy_dict else None)
                for v in DATA_Generation.keys()
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()
    
    @staticmethod
    def from_str(s: str) -> DATA_Generation:
        keys = DATA_Generation.keys()
        val_dict = {key: None for key in keys}
        s = [parse_string_val(val) for val in s.split("\t")]
        for i in range(len(keys)):
            if i >= len(s):
                break
            key = keys[i]
            val = s[i]

            if key == "watermark" and val is not None:
                val = DATA_WatermarkSpec.from_str(val)
            elif key in ["key", "token_count", "id"] and val is not None:
                val = int(val)
            elif (
                key
                in [
                    "rating",
                    "pvalue",
                    "efficiency",
                    "entropy",
                    "spike_entropy",
                ]
                and val is not None
            ):
                val = float(val)
            elif (key == "response" or key == "prompt") and val is not None:
                val = (
                    re.sub(r"(___LINE___)+$", "___LINE___", val)
                    .replace("___LINE___", "\n")
                    .replace("___TAB___", "\t")
                )

            val_dict[key] = val

        if val_dict["key"] is not None and val_dict["watermark"] is not None:
            val_dict["watermark"] = replace(
                val_dict["watermark"], secret_key=val_dict["key"]
            )

        if val_dict["temp"] is None and val_dict["watermark"] is not None:
            val_dict["temp"] = val_dict["watermark"].temp

        return DATA_Generation(**val_dict)
    
    @staticmethod
    def from_dict(d: Dict[str:Any]) -> DATA_Generation:
        return dacite.from_dict(DATA_Generation, d)
    @staticmethod
    def from_file(filename: str) -> List[DATA_Generation]:
        with open(filename, "r") as infile:
            raw = [l for l in infile.read().split("\n") if len(l)][1:]
        return [DATA_Generation.from_str(r) for r in raw]

    @staticmethod
    def to_file(
        filename: str, generations: Optional[List[DATA_Generation]] = None
    ) -> None:
        if not path.isfile(filename):
            with open(filename, "w") as outfile:
                outfile.write("\t".join(DATA_Generation.keys()) + "\n")
                if generations is not None and len(generations):
                    outfile.write("\n".join(str(g) for g in generations) + "\n")
        else:
            with open(filename, "a") as outfile:
                if generations is not None and len(generations):
                    outfile.write("\n".join(str(g) for g in generations) + "\n")

@dataclass
class DATA_Watermark:
    mode: str
    bench: str
    tokenizer: str
    gen_len: int
    param1: int_or_float
    param2: int_or_float
    ngram:int
    temperature: float
    seed: int
    beam_chunk_size: Optional[int] = 0
    bcs: Optional[int] = 0

    beam_search = False

    def __post_init__(self):
        if self.bcs == 0 and self.beam_chunk_size !=0:
            self.bcs = self.beam_chunk_size
        elif self.bcs != 0 and self.beam_chunk_size ==0:
            self.beam_chunk_size = self.bcs