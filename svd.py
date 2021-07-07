import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from helper_functions import clean_messages
from metrics import SVD


vocab =  "5" # 2, 5, 10, 100
_len = "10" # for vocab 2, len 1 and 6. For vocab 5, len 10. For vocab 10 len 2 and 6. For vocab 100, len 2 and 6
vocab_size = int(vocab) 


# ===========================================================
# ======== LOAD DATA (NO CONTEXT AND CONTEXT FILES) ========= 


path = "data/Vocab" + vocab + "/Vocab" + vocab + "Len" + _len + "Target"
df = pd.read_csv(path + "/runs/interactions.csv")

path_context = path + "Context"
df_context = pd.read_csv(path_context + "/runs/interactions.csv")

clean_messages(df)
clean_messages(df_context)


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
plt.savefig("svd_plots/SVD_vocab_"+vocab+"_max_len_"+_len+".png", bbox_inches="tight")
