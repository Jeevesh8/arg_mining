import os

def get_model_tok_version(args):
    tokenizer_dir = os.path.join(args.pretrained, "tokenizer")
    model_dir =  os.path.join(args.pretrained, "model")
    if os.path.isdir(model_dir) and os.path.isdir(tokenizer_dir):
        print("Using existing model at:", model_dir)
        print("Using existing Tokenizer at:", tokenizer_dir)
        return (tokenizer_dir, model_dir)
    elif ((os.path.isdir(model_dir) and not os.path.isdir(tokenizer_dir)) or
          (os.path.isdir(tokenizer_dir) and not os.path.isdir(model_dir))):
        raise ValueError("Both tokenizer and model directories must exist. Only one exists.")
    else:
        return (args.pretrained, args.pretrained)
