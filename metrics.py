import numpy as np
from helper_functions import unique_images, unique_messages

def average_message_length(df):

    """
    Average number of tokens in the messages produced by the Sender.
    """

    avg_len = sum(len(df.loc[row, "Message Modified"]) for row in range(len(df)))
    avg_len = avg_len/len(df)

    return avg_len


def perplexity_per_symbol(df, vocab_len):

    """
    (Not too much sense with only two symbols, but implemented anyway)
    Count how often a specific symbol is used to speak about a certain class object
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

def message_distinctness(df):
    """
    Metric for determining how much of 
    an image is captured in a message
    """
    
    # count of unique messages per batch
    # divided by the batch size
    
    print(f"{len(unique_messages(df))}")
    print(f"{len(df)}")
    
    md = len(unique_messages(df)) / len(df)
    
    # can also use distinct images in set as 
    # reference point
    comparison = md - len(unique_images(df))
    
    return md, comparison

def from_messages_to_categories(df):

    """
    Needed for PURITY metric
    Computes the number of times that each category is the true class 
    each time a specific message is sent.
    Example: fmtc['1110'] = {'tv': 6, 'person': 125, 'bottle': 16, 'tie': 6, ...}
    """

    fmtc = {}
    for row in range(len(df)):
        if df["Message Modified"][row] not in fmtc.keys():
            fmtc[df["Message Modified"][row]] = {}
        if df["True Class"][row] not in fmtc[df["Message Modified"][row]].keys():
            fmtc[df["Message Modified"][row]][df["True Class"][row]] = 1
        else:         
            fmtc[df["Message Modified"][row]][df["True Class"][row]] += 1

    return fmtc



def purity(df):

    """
    The purity of a clustering solution is the proportion of category labels
    in the clusters that agree with the respective cluster majority category.

    """
    catmex = from_messages_to_categories(df)

    for key, _ in catmex.items():
        #print("key=", key)
        max_val = max(catmex[key].values())
        max_key = max(catmex[key], key=catmex[key].get)
        #print(max_key, max_val)



def learned_classes(df):

    """
    Dict containing the number of times each true class has been corretly predicted by the receiver
    Example: correct_pred_perc['tv'] = 0.83 (8 times over 10 the tv class has been correctly predicted)
    """

    correct_pred_perc = {}; count = {}
    for row in range(len(df)):
        _class = df["True Class"][row]
        if (_class not in count.keys() ):
            count[_class] = 1
        else: 
            count[_class] += 1
        
        if (_class not in correct_pred_perc.keys() ):
            correct_pred_perc[_class] = 0.

        if (df["Is correct"][row] == True): 
            correct_pred_perc [_class] += 1.

    for _class in correct_pred_perc.keys():
        correct_pred_perc[_class] = correct_pred_perc[_class]/count[_class]
    
    return correct_pred_perc


def SVD(df, vocab_len):

    """
    Singular value decomposition for each couple of (true class, distractor class)
    """

    couples_dict = {}
    symbols = [str(i) for i in range(vocab_len)]
    for row in range(len(df)):

        couple = (df["True Class"][row], df["Distractor class"][row])
        if couple not in couples_dict.keys():
            couples_dict[couple] = dict.fromkeys([str(i) for i in range(vocab_len)], 0)
        
        for symbol in symbols:
            couples_dict[couple][symbol] += df["Message Modified"][row].count(symbol)
        
    mat = np.array([list(val.values()) for val in couples_dict.values()])

    return np.linalg.svd(mat, full_matrices=True)


def SVD_messages(df, messages):

    """

    """

    couples_dict = {}
    for row in range(len(df)):

        couple = (df["True Class"][row], df["Distractor class"][row])
        if couple not in couples_dict.keys():
            couples_dict[couple] = dict.fromkeys([mex for mex in messages], 0)
        
        for mex in messages:
            couples_dict[couple][mex] += df["Message Modified"][row].count(mex)
        
    #print("Couples dict=",couples_dict)

    mat = np.array([list(val.values()) for val in couples_dict.values()])
    print("mat=",mat)

    u, s, vh = np.linalg.svd(mat, full_matrices=True)
    print(u.shape, s.shape, vh.shape)
    print(s)