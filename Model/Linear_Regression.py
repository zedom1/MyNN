import np as np

class Linear_Regression(Model):
	def __init__(self, arg):
		
		self.input_dim = arg["input_dim"]
		self.output_dim = arg["output_dim"]
		self.w = np.random.normal(1, 1, size=(self.input_dim + 1, self.output_dim))
		self.updateMethod = arg["update"].lower() if "update" in arg else "sgd"

	def train(self, X, y, learning_rate = 0.01, update_size = None):

		assert len(np.shape(X)) == 2, "Invalid input shape. The input must be 2d (batch, input_dim)"
		batch_size, input_dim = np.shape(X)
		assert self.input_dim == input_dim, "Unmatch input_dim."


		X = np.column_stack((X, np.ones((batch_size, 1))))
		y = y.reshape((-1, self.output_dim))
		
		if self.updateMethod == "closed-form":
			# check whether the matrix can be inversed
			assert np.linalg.det(X) != 0, "This matrix cann't be inversed."
			
			# update the parameters
			self.w = np.linalg.inv(np.dot(X.T, X))  
		    self.w = np.dot(self.w, X.T)
		    self.w = np.dot(self.w, y)

		    # predict under the train set
		    Y_predict = np.dot(X, self.w)  
		    # calculate the absolute differences
		    loss_train = np.average(np.abs(Y_predict - y))  

		    return Y_predict, loss_train

		elif self.updateMethod.lower == "sgd":
			if update_size == None:
				update_size = batch_size
			assert update_size <= batch_size, "Update size lager than batch size!"
			y_predict = np.dot(X, self.w)
			diff = y_predict - y
		    randind = np.random.randint(0,X.shape[0]-1, size=update_size)

		    # calculate the gradient
		    G = -np.dot(X[randind].T.reshape(-1, update_size), y[randind].reshape(update_size, -1))   
		    G += np.dot(X[randind].T.reshape(-1, update_size), X[randind].reshape(update_size, -1)).dot(self.w)
		    G = -G
		    # update the parameters
		    self.w += learning_rate * G  

		    y_predict_selected = np.dot(X[randind], self.w)  
		    loss_train = np.average(np.abs(y_predict_selected - y[randind])) 
		    
		    return y_predict, loss_train

	def predict(self, X):
		assert len(np.shape(X)) == 2, "Invalid input shape. The input must be 2d (batch, input_dim)"
		batch_size, input_dim = np.shape(X)
		assert self.input_dim == input_dim, "Unmatch input_dim."

		X = np.column_stack((X, np.ones((batch_size, 1))))
		y_predict = np.dot(X, self.w)
		return y_predict
    