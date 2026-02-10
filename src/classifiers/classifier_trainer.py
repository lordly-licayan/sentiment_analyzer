from abc import ABC, abstractmethod


class ClassifierTrainer(ABC):
    @abstractmethod
    def train(self, job_id, X, y, comments, test_size=0.2, random_state=42):
        pass

    @abstractmethod
    def get_classifier(self):
        pass
