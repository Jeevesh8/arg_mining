def print_preds(tokenized_threads, tokenizer, preds, refs):
    for i, (thread, pred, ref) in enumerate(zip(tokenized_threads, preds, refs)):
        print("Thread: ", thread)
        print("Pred: ", pred)
        print("Ref: ", ref)
        print("Thread Text:", tokenizer.decode(thread))
        print("Predicted Components:")
        i=0
        while i<thread.shape[0] and thread[i]!=tokenizer.pad_token_id:
            if pred[i]=="B-C":
                start_idx = i
                i += 1
                while i<len(pred) and pred[i]=="I-C":
                    i += 1
                end_idx = i
                print("\tClaim:(", start_idx, end_idx, ")", tokenizer.decode(thread[start_idx:end_idx]))
            elif pred[i]=="B-P":
                start_idx = i
                i+=1
                while i<len(pred) and pred[i]=="I-P":
                    i += 1
                end_idx = i
                print("\tPremise:(", start_idx, end_idx, ")", tokenizer.decode(thread[start_idx:end_idx]))
            else:
                i += 1
        
        print("Actual Components:")
        i=0
        while i<thread.shape[0] and thread[i]!=tokenizer.pad_token_id:
            if ref[i]=="B-C":
                start_idx = i
                i += 1
                while i<len(ref) and ref[i]=="I-C":
                    i += 1
                end_idx = i
                print("\tClaim(", start_idx, end_idx, ")", tokenizer.decode(thread[start_idx:end_idx]))
            elif ref[i]=="B-P":
                start_idx = i
                i+=1
                while i<len(ref) and ref[i]=="I-P":
                    i += 1
                end_idx = i
                print("\tPremise:(", start_idx, end_idx, ")", tokenizer.decode(thread[start_idx:end_idx]))
            else:
                i += 1
            