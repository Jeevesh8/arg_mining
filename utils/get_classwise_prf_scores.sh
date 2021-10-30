REGEX="'C':.*?'precision': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 1

REGEX="'C':.*?'recall': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 1

REGEX="'C':.*?'f1': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 1

REGEX="'P':.*?'precision': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 1

REGEX="'P':.*?'recall': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 1

REGEX="'P':.*?'f1': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 1

REGEX="'overall_f1': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 1
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 1

echo "-----------------50-50 split-------------"
REGEX="'C':.*?'precision': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 2

REGEX="'C':.*?'recall': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 2

REGEX="'C':.*?'f1': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 2

REGEX="'P':.*?'precision': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 2

REGEX="'P':.*?'recall': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 2

REGEX="'P':.*?'f1': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 2

REGEX="'overall_f1': 0.(\d+)"
echo $REGEX
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 2
python3 last_n_means.py --in_files ../logs/out_5cons_runs_da_bert_ckpt4 --names "da-bert" --regexp "$REGEX" --split 2


#REGEX="'precision':.*?'parts_of_same': 0.(\d+)"
#echo $REGEX
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_base_lf --names "base Longformer prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_ckpt4 --names "sMLM ckpt4 prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_base_lf --names "base Longformer prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_ckpt4 --names "sMLM ckpt4 prompting 50-50" --regexp "$REGEX" --split 2

#REGEX="'recall':.*?'parts_of_same': 0.(\d+)"
#echo $REGEX
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_base_lf --names "base Longformer prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_ckpt4 --names "sMLM ckpt4 prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_base_lf --names "base Longformer prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_ckpt4 --names "sMLM ckpt4 prompting 50-50" --regexp "$REGEX" --split 2

#REGEX="'f1':.*?'parts_of_same': 0.(\d+)"
#echo $REGEX
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_base_lf --names "base Longformer prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_ckpt4 --names "sMLM ckpt4 prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_base_lf --names "base Longformer prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/DrInv_out_prompt_rel_pred_ckpt4 --names "sMLM ckpt4 prompting 50-50" --regexp "$REGEX" --split 2

#REGEX="'precision':.*?'partial': 0.(\d+)"
#echo $REGEX
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_base_lf --names "base Longformer prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_ckpt_4 --names "sMLM ckpt4 prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_base_lf --names "base Longformer prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_ckpt_4 --names "sMLM ckpt4 prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_base_lf --names "base Longformer mean pooling 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_ckpt4 --names "sMLM ckpt4 mean pooling 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_base_lf --names "base Longformer mean pooling 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_ckpt4 --names "sMLM ckpt4 mean pooling 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_roberta_CmvModes --names "Contextless Roberta 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_QRbert_CmvModes --names "Contextless QR-Bert 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_roberta_CmvModes --names "Contextless Roberta 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_QRbert_CmvModes --names "Contextless QR-Bert 50-50" --regexp "$REGEX" --split 2


#REGEX="'recall':.*?'partial': 0.(\d+)"
#echo $REGEX
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_base_lf --names "base Longformer prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_ckpt_4 --names "sMLM ckpt4 prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_base_lf --names "base Longformer prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_ckpt_4 --names "sMLM ckpt4 prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_base_lf --names "base Longformer mean pooling 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_ckpt4 --names "sMLM ckpt4 mean pooling 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_base_lf --names "base Longformer mean pooling 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_ckpt4 --names "sMLM ckpt4 mean pooling 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_roberta_CmvModes --names "Contextless Roberta 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_QRbert_CmvModes --names "Contextless QR-Bert 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_roberta_CmvModes --names "Contextless Roberta 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_QRbert_CmvModes --names "Contextless QR-Bert 50-50" --regexp "$REGEX" --split 2


#REGEX="'f1':.*?'partial': 0.(\d+)"
#echo $REGEX
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_base_lf --names "base Longformer prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_ckpt_4 --names "sMLM ckpt4 prompting 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_base_lf --names "base Longformer prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_prompt_rel_pred_ckpt_4 --names "sMLM ckpt4 prompting 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_base_lf --names "base Longformer mean pooling 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_ckpt4 --names "sMLM ckpt4 mean pooling 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_base_lf --names "base Longformer mean pooling 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_5cons_runs_ckpt4 --names "sMLM ckpt4 mean pooling 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_roberta_CmvModes --names "Contextless Roberta 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_QRbert_CmvModes --names "Contextless QR-Bert 80-20" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_roberta_CmvModes --names "Contextless Roberta 50-50" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_rel_pred_contextless_QRbert_CmvModes --names "Contextless QR-Bert 50-50" --regexp "$REGEX" --split 2


#REGEX="'D':.*?'f1': 0.(\d+)"
#echo $REGEX
#python3 last_n_means.py --in_files ../logs/out_5cons_runs_base_lf_drInventor --names "base Longformer" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_5cons_runs_ckpt_4_drInventor --names "sMLM ckpt4" --regexp "$REGEX" --split 1
#python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_base_cased --names "bert-base-cased" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_bert_ckpt4 --names "bert sMLM ckpt-4" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_5cons_runs_cw_base_roberta --names "Roberta" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_cw_5cons_runs_base_lf --names "Base LF" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_5cons_runs_base_lf_CmvModes_Global  --names "Base LF" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_5cons_runs_base_lf_256attWindow  --names "Base LF-256 attention window" --regexp "$REGEX" --split 2
#python3 last_n_means.py --in_files ../logs/out_5cons_runs_ckpt_4_CmvModes  --names "LF-ckpt4" --regexp "$REGEX" --split 2
