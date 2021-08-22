from flax.core.frozen_dict import freeze

#General Config
config = {
    "batch_size": 8,
    "num_devices": 1,
    "max_len": 4096,
    "max_comps": 128,
    "omit_filenames": True,

}

#Padding Config
config["pad_for"] = {
    "tokenized_thread": None,           #If None, set to tokenizer.pad_token_id when calling load_dataset()
    "comp_type_labels":
    config["arg_components"]["other"],  # len(config['arg_components']),
    "refers_to_and_type" : 0,  # len(config['dist_to_label'])+2,
}

#Data Representation Specific Config
config.update({
    "arg_components": {
        "other": 0,
        "B-C": 1,
        "I-C": 2,
        "B-P": 3,
        "I-P": 4
    },
    "relations": [
        "partial_attack",
        "agreement",
        "attack",
        "rebuttal_attack",
        "understand",
        "undercutter",
        "undercutter_attack",
        "disagreement",
        "rebuttal",
        "support",
        "None",
        "partial_agreement",
        "partial_disagreement",
    ],
    "relations_map": {
        "None": ["None"],
        "support": ["agreement", "understand", "support", "partial_agreement"],
        "against": [
            "partial_attack",
            "attack",
            "rebuttal_attack",
            "undercutter",
            "undercutter_attack",
            "disagreement",
            "rebuttal",
            "partial_disagreement",
        ],
    },
})

config = freeze(config)
