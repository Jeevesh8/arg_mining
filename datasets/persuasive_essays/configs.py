from flax.core.frozen_dict import freeze
from ..configs import tokenizer

config = {'arg_components': 
          {"other": 0,
           "B-C": 1,
           "I-C": 2,
           "B-P": 3,
           "I-P": 4},
          "max_len": 4096}

config["pad_for"] = {
    "tokenized_essays": tokenizer.pad_token_id,
    "comp_type_labels": config["arg_components"]["other"],
}

config = freeze(config)
