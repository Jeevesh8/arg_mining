from typing import List, Tuple
import re

from spacy.lang.en import English

class krip_alpha():
    """A module for computing sentence level Krippendorff's Alpha,
    for argumentative components  annotated at the token level. Must use
    labels ["B-C", "B-P"].
    """
    def __init__(self, tokenizer)-> None:
        """See self.compute_metric() for what each of these data actually mean.
        """
        self.tokenizer = tokenizer
        self.special_token_ids = ([tokenizer.convert_tokens_to_ids(elem) for elem in tokenizer.special_tokens_map.values()]+
                                  [elem for elem in tokenizer.get_added_vocab().values()])
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')

        self.pred_has_claim = 0
        self.ref_has_claim = 0
        self.pred_has_premise = 0
        self.ref_has_premise = 0
        
        self.claim_wise_agreement = 0
        self.premise_wise_agreement = 0
        
        self.claim_wise_disagreement = 0
        self.premise_wise_disagreement = 0
    
        self.total_sentences = 0
        
        self.has_both_ref = 0
        self.has_both_pred = 0
        self.has_none_ref = 0
        self.has_none_pred = 0

    def preprocess(self, threads: List[List[int]]) -> List[List[List[int]]]:
        """
        Args:
            threads:    A list of all threads in a batch. A thread is a list of 
                        integers corresponding to token_ids of the tokens in the 
                        thread.
        Returns:
            A List with all the threads, where each thread now consists of 
            sentence lists. Where, a sentence list in a thread list is the list 
            of token_ids corresponding to a sentence in a thread. 
        """
        threads_lis = []
        for i, thread in enumerate(threads):
            threads_lis.append([])
            thread = [elem for elem in thread if elem!=self.tokenizer.pad_token_id]
            txt_thread = self.tokenizer.decode(thread)
#            print("txt thread:", txt_thread)
            sentences = [sent.text for sent in self.nlp(txt_thread).sents]
            sentence_no, char_no = 0, 0
            prev_idx = 0
            for j, token in enumerate(thread):
                txt_token = self.tokenizer.convert_ids_to_tokens(token)
#                print("Txt token:", txt_token)
                txt_token = txt_token.replace("Ġ", " ")
                if txt_token.startswith("##"):
                    txt_token = txt_token[2:]
                for ch in txt_token:
                    if char_no>=len(sentences[sentence_no]):
                        threads_lis[-1].append(thread[prev_idx:j])
                        prev_idx = j
                        char_no = 0
                        sentence_no += 1
                    
                    if ch==sentences[sentence_no][char_no]:
                        char_no += 1
                    elif ch==" ":
                        continue
                    elif sentences[sentence_no][char_no]==" ":
                        while char_no<len(sentences[sentence_no]) and sentences[sentence_no][char_no]==" ":
                            char_no += 1
                        if ch==sentences[sentence_no][char_no]:
                            char_no += 1
                        else:
                            raise ValueError("Mismatch even after adjesting spaces")
                    else:
                        raise ValueError("Mismatch between sentences obtained from tokenizer.decode() token "+
                                txt_token+" and spacy sentence: "+sentences[sentence_no]+" .Unmatched characters:("+ch+","+
                                sentences[sentence_no][char_no]+")")

                if token in self.special_token_ids:
                    while char_no<len(sentences[sentence_no]) and sentences[sentence_no][char_no]==" ":
                        char_no += 1

            if prev_idx<len(thread):
                threads_lis[-1].append(thread[prev_idx:])
#            for sent in threads_lis[-1]:
#                print("Sentence:", self.tokenizer.decode(sent))

        return threads_lis

    def get_sentence_wise_preds(self, threads: List[List[List[int]]], 
                                      predictions: List[List[str]]) -> List[List[List[str]]]:
        """Splits the prediction corresponding to each thread, into predictions
        for each sentence in the corresponding thread in "threads" list.
        Args:
            threads:      A list of threads, where each thread consists of further 
                          lists corresponding to the various sentences in the
                          thread. [As output by self.preprocess()]
            predictions:  A list of predictions for each thread, in the threads
                          list. Each prediciton consists of a list of componenet 
                          types corresponding to each token in a thread.
        Returns:
            The predictions list, with each prediction split into predictions 
            corresponding to the sentences in the corresponding thread specified
            in the threads list. 
        """
        sentence_wise_preds = []
        for i, thread in enumerate(threads):
            next_sentence_beg = 0
            sentence_wise_preds.append([])
            for sentence in thread:
                sentence_wise_preds[i].append(
                    predictions[i][next_sentence_beg:next_sentence_beg+len(sentence)])
                next_sentence_beg += len(sentence)
        return sentence_wise_preds
    
    def update_state(self, pred_sentence: List[str], ref_sentence: List[str]) -> None:
        """Updates the various information maintained for the computation of
        Krippendorff's alpha, based on the predictions(pred_sentence) and 
        references(ref_sentence) provided for a particular sentence, in some 
        thread.
        """
        if len(pred_sentence)<=1:
            return

        self.total_sentences += 1
        
        if 'B-C' in pred_sentence:
            self.pred_has_claim += 1
            if 'B-C' in ref_sentence:
                self.ref_has_claim += 1
                self.claim_wise_agreement += 1
            else:
                self.claim_wise_disagreement += 1
            
        elif 'B-C' in ref_sentence:
            self.ref_has_claim += 1
            self.claim_wise_disagreement += 1
        
        else:
            self.claim_wise_agreement += 1
        
        if 'B-P' in pred_sentence:
            self.pred_has_premise += 1
            if 'B-P' in ref_sentence:
                self.ref_has_premise += 1
                self.premise_wise_agreement += 1
            else:
                self.premise_wise_disagreement += 1

        elif 'B-P' in ref_sentence:
            self.ref_has_premise += 1
            self.premise_wise_disagreement += 1
        
        else:
            self.premise_wise_agreement += 1
        
        if 'B-C' in pred_sentence and 'B-P' in pred_sentence:
            self.has_both_pred += 1
        
        if 'B-C' in ref_sentence and 'B-P' in ref_sentence:
            self.has_both_ref += 1
        
        if 'B-C' not in pred_sentence and 'B-P' not in pred_sentence:
            self.has_none_pred += 1
        
        if 'B-C' not in ref_sentence and 'B-P' not in ref_sentence:
            self.has_none_ref += 1
        return

    def add_batch(self, predictions: List[List[str]], 
                  references: List[List[str]], 
                  tokenized_threads: List[List[int]]) -> None:
        """Add a batch of predictions and references for the computation of 
        Krippendorff's alpha.
        Args:
            predictions:      A list of predictions for each thread, in the 
                              threads list. Each prediciton consists of a list 
                              of component types corresponding to each token in 
                              a thread.
            references:       Same structure as predictions, but consisting of 
                              acutal gold labels, instead of predicted ones.
            tokenized_thread: A list of all threads in a batch. A thread is a 
                              list of integers corresponding to token_ids of the
                              tokens in the thread.
        """
        threads = self.preprocess(tokenized_threads)
        
        sentence_wise_preds = self.get_sentence_wise_preds(threads, predictions)
        sentence_wise_refs = self.get_sentence_wise_preds(threads, references)

        for pred_thread, ref_thread in zip(sentence_wise_preds, sentence_wise_refs):
            for pred_sentence, ref_sentence in zip(pred_thread, ref_thread):
                self.update_state(pred_sentence, ref_sentence)

    def compute(self, print_additional: bool=True) -> None:
        """Prints out the metric, for the batched added till now. And then 
        resets all data being maintained by the metric. 
        Args:
            print_additional:   If True, will print all the data being 
                                maintained instead of just the Krippendorff's 
                                alphas for claims and premises.
        """
        print("Sentence level Krippendorff's alpha for Claims: ", 1-(self.claim_wise_disagreement/(self.claim_wise_agreement+self.claim_wise_disagreement))/0.5)
        print("Sentence level Krippendorff's alpha for Premises: ", 1-(self.premise_wise_disagreement/(self.premise_wise_agreement+self.premise_wise_disagreement))/0.5)
        
        if print_additional:
            print("Additional attributes: ")
            print("\tTotal Sentences:", self.total_sentences)
            print("\tPrediction setences having claims:", self.pred_has_claim)
            print("\tPrediction sentences having premises:", self.pred_has_premise)
            print("\tReference setences having claims:", self.ref_has_claim)
            print("\tReference sentences having premises:", self.ref_has_premise)
            print("\n")
            print("\tPrediction Sentence having both claim and premise:", self.has_both_pred)
            print("\tPrediction Sentence having neither claim nor premise:", self.has_none_pred)
            print("\tReference Sentence having both claim and premise:", self.has_both_ref)
            print("\tReference Sentence having neither claim nor premise:", self.has_none_ref)
            print("\n")
            print("\tSentences having claim in both reference and prediction:", self.claim_wise_agreement)
            print("\tSentences having claim in only one of reference or prediction:", self.claim_wise_disagreement)
            print("\tSentences having premise in both reference and prediction:", self.premise_wise_agreement)
            print("\tSentences having premise in only one of reference or prediction:", self.premise_wise_disagreement)
        self.__init__(self.tokenizer)
