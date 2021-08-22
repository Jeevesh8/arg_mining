from flax.core.frozen_dict import freeze

config = {
    'max_comps' : 128,
    'embed_dim' : 768,
}

config = freeze(config)