import copy, numpy as np
from activation import Sigmoid

np.random.seed(0)

def bin2int(d):
	out = 0
	for index,x in enumerate(d):
		out += x*pow(2,index)
	return out

class RNN(object):

	def __init__(self, arg):
		super(RNN, self).__init__()

		self.input_dim = arg["input_dim"]
		self.hidden_dim = arg["hidden_dim"]
		self.output_dim = arg["output_dim"]

		self.w_input_hidden = 2*np.random.random((self.input_dim, self.hidden_dim)) - 1
		self.w_hidden_output = 2*np.random.random((self.hidden_dim, self.output_dim)) - 1
		self.w_hidden_hidden = 2*np.random.random((self.hidden_dim, self.hidden_dim)) - 1

		self.activation = Sigmoid()

	def train(self, a, b, c, learning_rate = 0.1, verbose = False):

		d = np.zeros_like(c)
		overallError = 0
		deltas = list()
		predicts = list()
		hidden_layer = list()
		hidden_layer.append(np.zeros(self.hidden_dim))
		# Forwarding
		# moving along the positions in the binary encoding
		steps = np.shape(a)[1]

		for position in range(steps):
			# generate input and output
			
			X = np.array([a[0][position],b[0][position]]).T
			y = np.array([c[0][position]]).T

			# hidden layer (input ~+ pre_hidden)
			new_hidden = self.activation._calculate((np.dot(X, self.w_input_hidden) + np.dot(hidden_layer[-1], self.w_hidden_hidden)))

			# output layer (new binary representation)
			predict = self.activation._calculate(np.dot(new_hidden, self.w_hidden_output))
			predicts.append(predict)
			# did we miss?... if so, by how much?
			error = y - predict
			deltas.append((error)*self.activation._derivative_with_calculate(predict))
			overallError += np.abs(error[0][0])
			
			# store hidden layer so we can use it in the next timestep
			hidden_layer.append(copy.deepcopy(new_hidden))

		# Backward update initialize
		accumulate_hidden_delta = np.zeros(hidden_dim)

		w_ih_update = np.zeros_like(self.w_input_hidden)
		w_ho_update = np.zeros_like(self.w_hidden_output)
		w_hh_update = np.zeros_like(self.w_hidden_hidden)

		# Backward update
		for position in range(steps):
			
			X = np.array([a[0][steps - position - 1],b[0][steps - position - 1]]).T
			now_hidden_layer = hidden_layer[-position-1]
			pre_hidden_layer = np.atleast_2d(hidden_layer[-position-2])
			current_deltas = deltas[-position-1]
			
			delta_w_ho = now_hidden_layer.T.dot((np.atleast_2d(current_deltas)))
			w_ho_update += delta_w_ho

			delta_c_h = current_deltas.dot(self.w_hidden_output.T)
			
			delta_c_h += np.atleast_2d(accumulate_hidden_delta).dot(self.w_hidden_hidden)

			delta_c_h *= self.activation._derivative_with_calculate(now_hidden_layer)

			accumulate_hidden_delta = delta_c_h

			accumulate_derivative = accumulate_hidden_delta 
			w_ih_update += X.T.dot(accumulate_derivative)
			w_hh_update += pre_hidden_layer.T.dot(accumulate_derivative)

		self.w_input_hidden += w_ih_update * learning_rate
		self.w_hidden_hidden += w_hh_update * learning_rate
		self.w_hidden_output += w_ho_update * learning_rate 

		w_ih_update *= 0
		w_hh_update *= 0
		w_ho_update *= 0

		if verbose:
			print("Overall Error = {}".format(overallError))
		
		return predicts, hidden_layer[-1]
		
# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
	int2binary[i] = np.flip(binary[i], axis=0)

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

arguments = {
	"input_dim": input_dim, 
	"hidden_dim": hidden_dim, 
	"output_dim":output_dim
}

# training logic, for one epoch
acc = 0

epochs = 50000

rnn = RNN(arguments)

for i in range(epochs):
	# generate a simple addition problem (a + b = c)
	a_int = np.random.randint(largest_number/2) # int version
	a = int2binary[a_int] # binary encoding
	a = np.reshape(a, (1, -1, 1))

	b_int = np.random.randint(largest_number/2) # int version
	b = int2binary[b_int] # binary encoding
	b = np.reshape(b, (1, -1, 1))
	# true answer
	c_int = a_int + b_int
	c = int2binary[c_int]
	c = np.reshape(c, (1, -1, 1))



	if i%1000 == 0:
		output, hidden = rnn.train(a, b, c, verbose = True)
	else:
		output, hidden = rnn.train(a, b, c)

	# decode output
	d = list()
	for j in output:
		d.append(int(np.round(j[0])))
	predict_num = bin2int(np.squeeze(d))
	acc += int(predict_num == c_int)
	
	if i%1000 == 0:
		print("Pred: " + ' '.join([str(i) for i in d]))
		print("True: " + ' '.join([str(i[0]) for i in c[0]]))
		
		a_int = bin2int(np.squeeze(a))
		b_int = bin2int(np.squeeze(b))
		out = bin2int(np.squeeze(d))

		print(str(a_int) + " + " + str(b_int) + " = " + str(out))
		print("------------")
		print("Accuracy = {}".format(acc/1000.0))
		acc = 0

