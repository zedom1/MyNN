from rnn import RNN
import copy, numpy as np

np.random.seed(0)
def bin2int(d):
	out = 0
	for index,x in enumerate(d):
		out += x*pow(2,index)
	return out

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
total_loss = 0

epochs = 50000

rnn = RNN(arguments)

for i in range(1, epochs+1):
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
	X = np.concatenate((a,b), axis= 2)
	y = c


	if i%1000 == 0:
		output, hidden, loss = rnn.train(X, y, verbose = True)
	else:
		output, hidden, loss = rnn.train(X, y)

	# decode output
	d = list()
	for j in output:
		d.append(int(np.round(j[0])))
	predict_num = bin2int(np.squeeze(d))
	acc += int(predict_num == c_int)
	total_loss += loss
	
	if i%1000 == 0:
		print("Pred: " + ' '.join([str(i) for i in d]))
		print("True: " + ' '.join([str(i[0]) for i in c[0]]))
		
		a_int = bin2int(np.squeeze(a))
		b_int = bin2int(np.squeeze(b))
		out = bin2int(np.squeeze(d))

		print(str(a_int) + " + " + str(b_int) + " = " + str(out))
		print("------------")
		print("Accuracy = {}".format(acc/1000.0))
		print("Total loss = {}".format(total_loss/1000.0))
		total_loss = 0
		acc = 0

