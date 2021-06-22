from numpy.random import random
import pandas as pd
import json
import numpy as np
from metrics import mex_per_class, unique_messages, clean_messages, mex_per_class_normalized, perplexity_per_symbol, unique_classes_and_superclasses, from_messages_to_categories, purity

vocab = "2"
_len = "6" # 1 e 6



# ===========================================================
# ======== LOAD DATA (NO CONTEXT AND CONTEXT FILES) ========= 



path = "Vocab" + vocab + "Len" + _len + "Target/"
df = pd.read_csv(path + "runs/interactions.csv")

path_context = "Vocab" + vocab + "Len" + _len + "TargetContext/"
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



# ====================================================
# ======== APPLY METRICS ON THE TWO DATASETS =========



classes, superclasses, classes_with_superclasses = unique_classes_and_superclasses(df)
#print(classes)
#print(superclasses)
#print(classes_with_superclasses)
clean_messages(df)
clean_messages(df_context)

# METRICS ON THE "NO CONTEXT" FILE

tot_messages = len(df)
um = unique_messages(df["Message Modified"])
print("\nUnique messages no context=", um)
mc = mex_per_class(df, um)
print("Count of Messages per class. mc[tv]=", mc['tv'])
mcn = mex_per_class_normalized(mc)
print("Messages per class normalized mcn[tv]=", mcn['tv'])

pps = perplexity_per_symbol(df, vocab_size)

catmex = from_messages_to_categories(df)
#print(catmex)
#purity(catmex)


# METRICS ON THE "CONTEXT" FILE

um_c = unique_messages(df_context["Message Modified"])
print("\nUnique messages context=", um_c)
mc_c = mex_per_class(df_context, um_c)
print("Count of Messages per class. mc_c[tv]=", mc_c['tv'])
mcn_c = mex_per_class_normalized(mc_c)
print("Messages per class normalized mcn_c[tv]=", mcn_c['tv'])

pps_c = perplexity_per_symbol(df_context, vocab_size)
catmex_c = from_messages_to_categories(df_context)
#purity(catmex_c)



# =========================================
# ======== SAVE METRICS IN DATASET ========



col_mex_count = ["mex-"+str(i)+"-c" for i in um.keys()] + ["mex-"+str(i)+"-no-c" for i in um.keys()] # no-c = no context
col_mex_perpl = ["pps-symbol-"+str(i)+"-c" for i in range(vocab_size)] + ["pps-symbol-"+str(i)+"-no-c" for i in range(vocab_size)] # pps = perplexity per symbol c=context

cols = col_mex_count + col_mex_perpl
df_analysis = pd.DataFrame(columns = ["true superclass"] + cols)

for _class in classes:
    df_analysis.loc[_class] = [classes_with_superclasses[_class]] + \
        [mc[_class][mex] for mex in um.keys()] + \
        [mc_c[_class][mex] for mex in um.keys()] + \
        [pps[_class][symbol] for symbol in range(vocab_size)] + \
        [pps_c[_class][symbol] for symbol in range(vocab_size)]

# avg e std of columns
col_numbers = [col for col in df_analysis.columns]
col_numbers.remove("true superclass")
df_analysis.loc["SUM COL"] = ["SUM COL"] + [df_analysis[col].sum() for col in col_numbers]
df_analysis.loc["MEAN COL"] = ["MEAN COL"] + [df_analysis[col].mean() for col in col_numbers]
df_analysis.loc["STD COL"] = ["STD COL"] + [df_analysis[col].std() for col in col_numbers]

df_analysis.to_csv("analysis_data_vocab"+vocab+"_len"+_len+".csv")



# ========================================
# ======== SUPERCLASSES ANALYSIS =========



df_super = pd.DataFrame(columns = cols)

for superclass in superclasses:
    df_tmp = df_analysis.loc[df_analysis['true superclass'] == superclass]
    df_super.loc[superclass] = [df_tmp[col].mean() for col in cols]

df_super.to_csv("analysis_superclasses_vocab"+vocab+"_len"+_len+".csv")