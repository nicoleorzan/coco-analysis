
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


def unique_images(df):
    """
    Returns a set of all distinct image
    ids from data frame
    (useful for message distinctness)

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ui = set(df["Image id"])
    return ui

def unique_messages(df):

    """
    List of the unique messages used by the sender and their counts. OK
    """

    um = {}

    for row in range(len(df)):
        #mex = df.loc[row, "Message Modified"]
        mex = df["Message Modified"][row]

        if (mex not in um.keys()):
            um[mex] = 1
        else:
            um[mex] += 1
        
    return um


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


def mex_per_class(df, um):

    """
    Step 1:
    Check how often a specific message is used to explain a certain goal class.
    Dictionary: key is the class that should be predicted, and value is the sent message.

    Example: if we have two messages, '10' and '0', we may get: 
    mc[tv] = {'10': 12, '0': 20}.

    Step 2: 
    Normalizes the above measure to get "probabilities" instead of counts

    Example: mpc_norm[tv] = {'10': 0.375, '0': 0.625}.
    """
    
    mpc = {}

    for row in range(len(df)):
        _class = df["True Class"][row]

        if _class not in mpc.keys():
            mpc[_class] = {key: 0 for key in um.keys()}

        mex = df.loc[row, "Message Modified"]
        mpc[ _class ][ mex ] += 1

    # NORMALIZATION OF THE ABOVE MEASURE

    mpc_norm = {}

    for concept in mpc.keys():
        denom = sum(mpc[concept].values(), 0.0)
        mpc_norm[concept]  = {k: v / denom for k, v in mpc[concept].items()}

    return mpc, mpc_norm


def mean_mex_per_class(mpc):

    """ 
    of messages per class / # classes
    """

    avg_mex = 0
    for _class in mpc.keys():
        avg_mex += len(mpc[_class].keys())
    avg_mex = avg_mex/len(mpc.keys())
    
    return avg_mex