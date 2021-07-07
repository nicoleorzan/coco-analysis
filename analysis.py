import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from helper_functions import mean_mex_per_class, mex_per_class, unique_messages, clean_messages, unique_classes_and_superclasses
from metrics import  perplexity_per_symbol, message_distinctness, SVD


vocab = "5" # 2, 5, 10, 100
_len = "10" # for vocab 2, len 1 and 6. For vocab 5, len 10. For vocab 10 len 2 and 6. For vocab 100, len 2 and 6
save_path = "df_analysis/"


# ===========================================================
# ======== LOAD DATA (NO CONTEXT AND CONTEXT FILES) ========= 



path = "data/Vocab" + vocab + "/Vocab" + vocab + "Len" + _len + "Target"
df = pd.read_csv(path + "/runs/interactions.csv")

path_context = path + "Context"
df_context = pd.read_csv(path_context + "/runs/interactions.csv")

with open(path + "/params.json") as file:
    json_file = json.load(file)

with open(path_context + "/params.json") as file:
    json_file_context = json.load(file)

# small check
assert(json_file['vocab_size'] == json_file_context['vocab_size'] )
assert(json_file['max_len'] == json_file_context['max_len'] )
assert(json_file['batch_size'] == json_file_context['batch_size'] )
vocab_size = json_file['vocab_size']



# ====================================================
# ======== APPLY METRICS ON THE TWO DATASETS =========


classes, superclasses, classes_with_superclasses = unique_classes_and_superclasses(df)


clean_messages(df)
clean_messages(df_context)


# ================================
# METRICS ON THE "NO CONTEXT" FILE

print(message_distinctness(df))

print("\nNO CONTEXT DATASET")

um = unique_messages(df)
print("* Unique messages=", um)
mpc, mpc_n = mex_per_class(df, um)
#print("* Count of Messages per class\n  Example: mpc[tv]=", mpc['tv'])
#print("* Messages per class normalized\n  Example: mpc_n[tv]=", mpc_n['tv'])
avg_mex = mean_mex_per_class(mpc)
print("* Average on the number of messages per class:", avg_mex)

pps = perplexity_per_symbol(df, vocab_size)
print(pps)


# =============================
# METRICS ON THE "CONTEXT" FILE


print("\nCONTEXT DATASET")

um_c = unique_messages(df_context)
print("* Unique messages=", um)
mpc_c, mpc_n_c = mex_per_class(df_context, um_c)
#print("* Count of Messages per class\n  Example: mpc_c[tv]=", mpc_c['tv'])
#print("* Messages per class normalized\n  Example: mpc_n_c[tv]=", mpc_n_c['tv'])
avg_mex_c = mean_mex_per_class(mpc_c)
print("* Average on the number of messages per class:", avg_mex_c)

pps_c = perplexity_per_symbol(df_context, vocab_size)
print(pps_c)


# =========================================
# ======== SAVE METRICS IN DATASET ========

print("\nSAVE METRICS IN DATASET")


col_mex_count = ["mex-"+str(i)+"-c" for i in um.keys()] + ["mex-"+str(i)+"-no-c" for i in um_c.keys()] # no-c = no context
col_mex_perpl = ["pps-symbol-"+str(i)+"-c" for i in range(vocab_size)] + ["pps-symbol-"+str(i)+"-no-c" for i in range(vocab_size)] # pps = perplexity per symbol c=context

cols = col_mex_count + col_mex_perpl
df_analysis = pd.DataFrame(columns = ["true superclass"] + cols)

for _class in classes:
    df_analysis.loc[_class] = [classes_with_superclasses[_class]] + \
        [mpc[_class][mex] for mex in um.keys()] + \
        [mpc_c[_class][mex] for mex in um_c.keys()] + \
        [pps[_class][symbol] for symbol in range(vocab_size)] + \
        [pps_c[_class][symbol] for symbol in range(vocab_size)]

# avg e std of columns
col_numbers = [col for col in df_analysis.columns]
col_numbers.remove("true superclass")
df_analysis.loc["SUM COL"] = ["SUM COL"] + [df_analysis[col].sum() for col in col_numbers]
df_analysis.loc["MEAN COL"] = ["MEAN COL"] + [df_analysis[col].mean() for col in col_numbers]
df_analysis.loc["STD COL"] = ["STD COL"] + [df_analysis[col].std() for col in col_numbers]

df_analysis.to_csv(save_path + "Vocab" + vocab + "/analysis_data_vocab" + vocab + "_len" + _len + ".csv")


# ========================================
# ======== SUPERCLASSES ANALYSIS =========


print("\nSUPERCLASSES ANALYSIS")

df_super = pd.DataFrame(columns = cols)

for superclass in superclasses:
    df_tmp = df_analysis.loc[df_analysis['true superclass'] == superclass]
    df_super.loc[superclass] = [df_tmp[col].sum() for col in cols]

df_super.to_csv(save_path + "Vocab" + vocab + "/analysis_superclasses_vocab" + vocab + "_len" + _len + ".csv")



# =========================================
# ===== SINGULAR VALUES DECOMPOSITION =====


print("SVD")
u, s, vh = SVD(df, vocab_size)
u_c, s_c, vh_c = SVD(df_context, vocab_size)

print("No context =", s)
print("Context    =", s_c)

plt.plot(np.linspace(0, len(s), len(s)), s/(np.sum(s)), label="no context", marker='o', markersize=4)
plt.plot(np.linspace(0, len(s_c), len(s_c)), s_c/(np.sum(s_c)), label="context", marker='o', markersize=4)
plt.title("Singular Value Decomposition for vocab "+vocab+", max_len "+_len)
plt.legend()
plt.grid()
plt.savefig("plots/SVD_vocab_"+vocab+"_max_len_"+_len+".png", bbox_inches="tight")
