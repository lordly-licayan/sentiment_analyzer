from sklearn.linear_model import SGDClassifier


def create_SGD_classifier():
    """
    Create a SGDClassifier with predefined parameters.
    """
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        eta0=0.001,
        max_iter=1,
        warm_start=True,
        tol=None,
        random_state=42,
    )

    return clf
