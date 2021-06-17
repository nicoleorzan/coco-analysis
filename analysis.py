import pandas as pd
import json
from metrics import mex_per_class, unique_messages, modif_messages, mex_per_class_normalized, perplexity_per_symbol, unique_classes

vocab = "2"
_len = "1" # 1 e 6


# ======== LOAD DATA  ========= 


path = "/home/nicole/Heavy_stuff/Analysis/Vocab"+vocab+"Len"+_len+"Target/"
df = pd.read_csv(path + "runs/interactions.csv")

path_context = "/home/nicole/Heavy_stuff/Analysis/Vocab"+vocab+"Len"+_len+"TargetContext/"
df_context = pd.read_csv(path_context + "runs/interactions.csv")

with open(path + "params.json") as file:
    json_file = json.load(file)

with open(path_context + "params.json") as file:
    json_file_context = json.load(file)


# small check
assert(json_file['vocab_size'] == json_file_context['vocab_size'] )
assert(json_file['max_len'] == json_file_context['max_len'] )
assert(json_file['batch_size'] == json_file_context['batch_size'] )
vocab_size = json_file['vocab_size']


# ======== APPLY METRICS =========


classes = unique_classes(df)
modif_messages(df)
modif_messages(df_context)

# NO CONTEXT

um = unique_messages(df)
print("um=", um)
mc = mex_per_class(df, um)
print("Mex per class. mc[tv]=", mc['tv'])
mcn = mex_per_class_normalized(mc)
print("Mex per class normalized mcn[tv]=", mcn['tv'])

pps = perplexity_per_symbol(df, vocab_size)

# CONTEXT 

um_c = unique_messages(df_context)
print("um_c=", um_c)
mc_c = mex_per_class(df_context, um_c)
print("Mex per class. mc_c[tv]=", mc_c['tv'])
mcn_c = mex_per_class_normalized(mc_c)
print("Mex per class normalized mcn_c[tv]=", mcn_c['tv'])

pps_c = perplexity_per_symbol(df_context, vocab_size)


# ======== SAVE METRICS IN DATASET ========


df_analysis = pd.DataFrame(columns = ["mex_per_class_0_no_context", "mex_per_class_0_context", "mex_per_class_10_no_context", "mex_per_class_10_context" ,"perplexity_no_context_symbol0", "perplexity_context_symbol0" , "perplexity_no_context_symbol1", "perplexity_context_symbol1"])

for _class in classes:
    df_analysis.loc[_class] = [mcn[_class]['0'], mcn_c[_class]['0'], mcn[_class]['10'], mcn_c[_class]['10'], pps[_class][0], pps_c[_class][0], pps[_class][1], pps_c[_class][1] ]

df_analysis.to_csv("analysis.csv")