from flax.core.frozen_dict import freeze

config = {'arg_components': 
          {"other": 0,
           "B-C": 1,
           "I-C": 2,
           "B-P": 3,
           "I-P": 4},
          "max_len": 512,
          "batch_size": 512*8}

config["pad_for"] = {
    "tokenized_essays": None,                               #Set to tokenizer.pad_token_id if None
    "comp_type_labels": config["arg_components"]["other"],
}

config = freeze(config)
