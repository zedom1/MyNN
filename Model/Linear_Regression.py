import numpy as np

n_samples, n_features = X.shape
X = np.column_stack((X, np.ones((n_samples, 1))))
y = y.reshape((-1, 1))

w = np.random.normal(1, 1, size=(n_features + 1, 1))

# closed form
for epoch in range(max_epoch):
    w = np.linalg.inv(np.dot(X_train.T, X_train))  # update the parameters
    w = np.dot(w, X_train.T)
    w = np.dot(w, y_train)
    Y_predict = np.dot(X_train, w)  # predict under the train set
    loss_train = np.average(np.abs(Y_predict - y_train))  # calculate the absolute differences
    losses_train.append(loss_train)

    Y_predict = np.dot(X_val, w)  # predict under the validation set
    rmse_val.append(rmse(Y_predict, y_val))
    loss_val = np.average(np.abs(Y_predict - y_val))  # calculate the absolute differences
    losses_val.append(loss_val)
    #loss_zeros.append(loss_val)
    #loss_random.append(loss_val)
    loss_normal.append(loss_val)



## sgd
for epoch in range(max_epoch):
    diff = numpy.dot(X_train, w) - y_train
    randind = numpy.random.randint(0,X_train.shape[0]-1)
    
    G = -numpy.dot(X_train[randind].T.reshape(-1,1), y_train[randind].reshape(-1,1))   # calculate the gradient
    
    G += numpy.dot(X_train[randind].T.reshape(-1,1),(X_train[randind].reshape(1,-1))).dot(w)
    G = -G
    w += learning_rate * G  # update the parameters

    Y_predict = numpy.dot(X_train[randind], w)  # predict under the train set
    loss_train = numpy.average(numpy.abs(Y_predict - y_train[randind]))  # calculate the absolute differences
    losses_train.append(loss_train)
    

    Y_predict = numpy.dot(X_val, w)  # predict under the validation set
    rmse_val.append(rmse(Y_predict, y_val))
    loss_val = numpy.average(numpy.abs(Y_predict - y_val))  # calculate the absolute differences
    losses_val.append(loss_val)
    #loss_zeros.append(loss_val)
    #loss_random.append(loss_val)
    loss_normal.append(loss_val)
    