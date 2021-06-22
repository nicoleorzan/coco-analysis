import numpy as np


def unique_classes_and_superclasses(df):

    """
    List of the unique classes that needs to be predicted by the receiver ("true class" in the dataset)
    """
    
    uc = []
    supercl = []
    classes = {}

    for row in range(len(df)):
        if (df.loc[row, "True SuperClass"] not in supercl):
            supercl.append(df.loc[row, "True SuperClass"])

        if (df.loc[row, "True Class"] not in classes.keys()):
            classes[df.loc[row, "True Class"] ] = df.loc[row, "True SuperClass"] 

        if (df.loc[row, "True Class"] not in uc):
           uc.append(df.loc[row, "True Class"])
        
    return uc, supercl, classes


def clean_messages(df):

    """
    Modify messages taking away ";" and takng the part from 0 to mex_length
    Adds a new column to the dataset. OK
    """

    mex_modif = []

    for row in range(len(df)):
        m_len = int( df.loc[row, "Message Length"] )
        mex = df.loc[row, "Message"].replace(';', '')[0:m_len]
        mex_modif.append(mex)
        
    df['Message Modified'] = mex_modif


def unique_messages(df):

    """
    List of the unique messages used by the sender and their counts. OK
    """

    um = {}

    for row in range(len(df)):
        #mex = df.loc[row, "Message Modified"]
        mex = df[row]

        if (mex not in um.keys()):
            um[mex] = 1
        else:
            um[mex] += 1
        
    return um


def mex_per_class(df, um):

    """
    Check how often a specific message is used to explain a certain goal class.
    Dictionary: key is the class that should be predicted, and value is the sent message.

    Example: if we have two messages, '10' and '0', we may get: 
    mc[tv] = {'10': 12, '0': 20}. OK

    """
    
    mpc = {}

    for row in range(len(df)):
        _class = df["True Class"][row]

        if _class not in mpc.keys():
            mpc[_class] = {key: 0 for key in um.keys()}

        mex = df.loc[row, "Message Modified"]
        mpc[ _class ][ mex ] += 1

    return mpc
    

def mex_per_class_normalized(mpc):

    """
    Normalizes the above measure to get "probabilities" instead of counts

    Example: mpc_norm[tv] = {'10': 0.375, '0': 0.625}. OK

    """
    
    mpc_norm = {}

    for concept in mpc.keys():
        denom = sum(mpc[concept].values(), 0.0)
        mpc_norm[concept]  = {k: v / denom for k, v in mpc[concept].items()}

    return mpc_norm


def average_message_length(df):

    """
    Average number of tokens in the messages produced by the Sender. OK
    """

    avg_len = sum(len(df.loc[row, "Message Modified"]) for row in range(len(df)))
    avg_len = avg_len/len(df)

    return avg_len


def perplexity_per_symbol(df, vocab_len):

    """
    (Not too much sense with only two symbols, but implemented anyway)
    Count how often a specific sybol is used to speak about a certain class object
    If vocab_len = 2, symbols are 0 and 1
    A low perplexity shows that the same symbols are consistently used to describe the same objects
    """
    pps = {}
    pps_norm = {}

    for row in range(len(df)):

        _class = df["True Class"][row]
        if _class not in pps.keys():
            pps[_class] = dict.fromkeys([i for i in range(vocab_len)], 0)

        mex = df["Message Modified"][row]
        
        for i in range(len(mex)):
            pps[_class][int(mex[i])] += 1
   
    return pps


# This metric should be computed during the TRAINING!!
def message_distinctness(df, batch_size):

    """
    Number of unique messages in each mini-batch (# unique messages / batch size)
    """

    mds = []

    for i, df_tmp in df.groupby(np.arange(len(df)) // batch_size):
        unique_messages = []

        for row_idx in range(0, batch_size):
            if (i*batch_size + row_idx >= len(df)):
                break

            if (df_tmp.loc[i*batch_size + row_idx, "Message"] not in unique_messages):
                unique_messages.append(df.loc[i*batch_size + row_idx, "Message"] )
        
            mex_dist = len(unique_messages)/batch_size
            mds.append(mex_dist)

    return unique_messages, np.mean(mds)


def from_messages_to_categories(df):

    dic = {}
    for row in range(len(df)):
        if df["Message Modified"][row] not in dic.keys():
            dic[df["Message Modified"][row]] = {}
        if df["True Class"][row] not in dic[df["Message Modified"][row]].keys():
            dic[df["Message Modified"][row]][df["True Class"][row]] = 1
        else:         
            dic[df["Message Modified"][row]][df["True Class"][row]] += 1

    return dic

def purity(catmex):

    """
    The purity of a clustering solution is the proportion of category labels
    in the clusters that agree with the respective cluster majority category.

    """
    for key, _ in catmex.items():
        print("key=", key)
        max_val = max(catmex[key].values())
        max_key = max(catmex[key], key=catmex[key].get)
        print(max_key, max_val)



