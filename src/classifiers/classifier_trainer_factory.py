from src.classifiers.classifier_trainer import ClassifierTrainer
from src.classifiers.logistic_regression import LogisticRegressionClassifier
from src.classifiers.sgd_classifier import SGDClassifierModel


class ClassifierTrainerFactory:
    _classifiers = {
        "LogisticRegression": LogisticRegressionClassifier,
        "SGDClassifier": SGDClassifierModel,
        # "SVMClassifier": SGDClassifierModel,
    }

    @staticmethod
    def get_classifier_trainer(
        classifier_name: str, model_name: str = None
    ) -> ClassifierTrainer:
        if classifier_name not in ClassifierTrainerFactory._classifiers:
            raise ValueError("Unsupported classifier.")
        return ClassifierTrainerFactory._classifiers[classifier_name](
            model_name=model_name
        )
