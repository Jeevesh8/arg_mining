from arg_mining.datasets.cmv_modes import data_config

ac_dict = data_config["arg_components"]

def print_preds(tokenized_threads, tokenizer, preds, refs):
    for i, (thread, pred, ref) in enumerate(zip(tokenized_threads, preds, refs)):
        print("Thread: ", thread)
        print("Pred: ", pred)
        print("Ref: ", ref)
        print("Thread Text:", tokenizer.decode(thread))
        print("Predicted Components:")
        i=0
        while i<=thread.shape[0] and thread[i]!=tokenizer.pad_token_id:
            if pred[i]==ac_dict["B-C"]:
                start_idx = i
                i += 1
                while pred[i]==ac_dict["I-C"]:
                    i += 1
                end_idx = i
                print("\t","Claim:",tokenizer.decode(thread[start_idx:end_idx]))
            elif pred[i]==ac_dict["B-P"]:
                start_idx = i
                while pred[i]==ac_dict["I-P"]:
                    i += 1
                end_idx = i
                print("\t", "Premise:", tokenizer.decode(thread[start_idx:end_idx]))
            else:
                i += 1
        
        print("Actual Components:")
        i=0
        while i<=thread.shape[0] and thread[i]!=tokenizer.pad_token_id:
            if ref[i]==ac_dict["B-C"]:
                start_idx = i
                i += 1
                while ref[i]==ac_dict["I-C"]:
                    i += 1
                end_idx = i
                print("\t", "Claim:", tokenizer.decode(thread[start_idx:end_idx]))
            elif ref[i]==ac_dict["B-P"]:
                start_idx = i
                while pred[i]==ac_dict["I-P"]:
                    i += 1
                end_idx = i
                print("\t", "Premise:", tokenizer.decode(thread[start_idx:end_idx]))
            else:
                i += 1
            