# Coco Game Analysis

Implemented metrics:
* Average Message Length: Average number of tokens in the messages produced by the Sender
* Perplexity per Symbol: Counts how often a specific sybol is used to speak about a certain object (If vocab_len = 2, symbols are 0 and 1)

Implemented helper functions:
* Unique Classes: List of the unique classes that needs to be predicted by the receiver ("true class" in the dataset)
* Clean Messages: Modifies the messages taking away the ";" and takng the part from 0 to mex_length. Adds a new column to the dataset.
* Unique Messages: List of the unique messages used by the sender and their counts.
* Messages per Class: Checks how often a specific message is used to explain a certain goal class. Creates a dictionary: key is the class that should be predicted, and value is the sent message (Example: if we have two messages, '10' and '0', we may get: mc['tv'] = {'10': 12, '0': 20}.
* Messages per Class normalized: Normalizes the above measure to get "probabilities" instead of counts (Example: mpc_norm['tv'] = {'10': 0.375, '0': 0.625})

