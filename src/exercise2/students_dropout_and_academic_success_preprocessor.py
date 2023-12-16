def preprocess(data):
    # extract X and y
    X = data.drop("Target", axis=1)
    y = data["Target"]

    return X, y
