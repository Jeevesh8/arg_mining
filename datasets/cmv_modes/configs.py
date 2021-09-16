from flax.core.frozen_dict import freeze

#General Config
config = {
    "batch_size": 2,
    "num_devices": 1,
    "max_len": 4096,
    "max_comps": 128,
    "omit_filenames": True,

    "arg_components": {
        "O": 0,
        "B-C": 1,
        "I-C": 2,
        "B-P": 3,
        "I-P": 4
    },

    "max_users" : 10,                                                                  #Extra users, above max_users, are marked as unknown user "[UNU]"
}

config["special_tokens"] = ["[STARTQ]", "[ENDQ]", "[URL]", "[NEWLINE]",                            #[STARTQ], [ENDQ] are put around sentences quoted from previous replies, not around anything within " "
                            "[UNU]"]+[f"[USER{i}]" for i in range(config["max_users"])]


#Padding Config
config["pad_for"] = {
    "tokenized_thread": None,           #If None, set to tokenizer.pad_token_id when calling load_dataset()
    "comp_type_labels":
    config["arg_components"]["O"],  # len(config['arg_components']),
    "refers_to_and_type" : 0,  # len(config['dist_to_label'])+2,
}

#Data Representation Specific Config
config.update({
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
        "partial_agreement",
        "partial_disagreement",
        "None",
    ],
    
    "reduce_relations" : False,

    "relations_map": {
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
        "None": ["None"],
    },
})

config = freeze(config)