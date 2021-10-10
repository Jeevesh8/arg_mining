from sklearn.metrics import precision_recall_fscore_support as prf_metric

class precision_recall_fscore():
    
    def __init__(self, tags_dict):
        self.preds = []
        self.refs = []
        self.tags_dict = tags_dict
    
    def add_batch(self, predictions, references):
        self.preds += predictions
        self.refs += references

    def compute(self):
        f1_metrics = {"precision" : {}, "recall" : {},
                      "f1": {}, "support": {}} 
        precision, recall, f1, supp = prf_metric(self.refs, self.preds, 
                                                 labels=list(self.tags_dict.keys()))

        for i, k in enumerate(self.tags_dict.keys()):
            f1_metrics["precision"][k] = precision[i]
            f1_metrics["recall"][k] = recall[i]
            f1_metrics["f1"][k] = f1[i]
            f1_metrics["support"][k] = supp[i]

        for avg in ["micro", "macro", "weighted"]:
            precision, recall, f1, supp = prf_metric(self.refs, self.preds,
                                                     labels=list(self.tags_dict.keys()), average=avg)
            f1_metrics[avg+"_avg"] = {}
            f1_metrics[avg+"_avg"]["precision"] = precision 
            f1_metrics[avg+"_avg"]["recall"] = recall
            f1_metrics[avg+"_avg"]["f1"] = f1
            f1_metrics[avg+"_avg"]["support"] = supp

        self.preds = []
        self.refs = []

        return f1_metrics