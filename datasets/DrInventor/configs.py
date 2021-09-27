from flax.core.frozen_dict import freeze

config = {"arg_components": {"O": 0,
                             "B-BC" : 1,
                             "I-BC" : 2,
                             "B-OC" : 3,
                             "I-OC" : 4,
                             "B-D"  : 5,
                             "I-D"  : 6,
                                },
          "rel_type_to_id":
          {
             "supports":0,
             "contradicts":1,
             "parts_of_same":2
          }
         }

config["pad_for"] = {"arg_components" : config["arg_components"]["O"],
                     "relations" : config["rel_type_to_id"]["supports"],}


config["special_tokens"] = ["[NEWLINE]"]

config = freeze(config)
