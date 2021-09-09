from flax.core.frozen_dict import freeze

config = {"arg_components": {"other": 0,
                             "B-BC" : 1,
                             "I-BC" : 2,
                             "B-OC" : 3,
                             "I-OC" : 4,
                             "B-D"  : 5,
                             "I-D"  : 6,
                                },
          "max_users": 10,
         }

config["pad_for"] = {"arg_components" : config["arg_components"]["other"],}

config["special_tokens"] = ["[NEWLINE]"]

config = freeze(config)
