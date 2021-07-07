import pandas as pd
from helper_functions import unique_messages, clean_messages, mean_mex_per_class, mex_per_class, clean_messages
from metrics import average_message_length

rootdir = 'data'

vocab_maxlen = {
    '2': ['1', '6'],
    '5': ['10'],
    '10': ['2', '6'],
    '100': ['2', '6']
}

general_cols = ["context?", "vocab", "max_len", "avg accuracy", "avg sender entropy", "avg loss", "avg mex len", "avg number different mex per class"]
df_general = pd.DataFrame(columns = general_cols)

i = 0
for vocab, max_lens in vocab_maxlen.items():
    print(vocab, max_lens)

    for _len in max_lens:

        path = "data/Vocab" + vocab + "/Vocab" + vocab + "Len" + _len + "Target"
        df = pd.read_csv(path + "/runs/interactions.csv")

        path_context = path + "Context"
        df_context = pd.read_csv(path_context + "/runs/interactions.csv")

        clean_messages(df)
        clean_messages(df_context)

        aml = average_message_length(df)
        um = unique_messages(df)
        mpc, _ = mex_per_class(df, um)
        avg_mex = mean_mex_per_class(mpc)

        aml_c = average_message_length(df_context)
        um_c = unique_messages(df_context)
        mpc_c, _ = mex_per_class(df_context, um_c)
        avg_mex_c = mean_mex_per_class(mpc_c)

        df_general.loc[i] = ["no", vocab, _len, df["Accuracy"].mean(), df["Sender Entropy"].mean(), df["Loss"].mean(), aml, avg_mex]
        df_general.loc[i+1] = ["yes", vocab, _len, df_context["Accuracy"].mean(), df_context["Sender Entropy"].mean(), df_context["Loss"].mean(), aml_c, avg_mex_c]
        i += 2


df_general.to_csv("analysis_general.csv")
        

