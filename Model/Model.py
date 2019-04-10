import abc

class Model(metaclass = abc.ABCMeta):
	def __init__(self):
		pass

	@abc.abstractmethod
	def train(self, X, y):
		pass

	@abc.abstractmethod
	def predict(self, X):
		pass
