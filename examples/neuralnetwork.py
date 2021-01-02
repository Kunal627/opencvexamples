import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        
        # no bias term required in out layer
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network
        # architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self,x):
        return 1.0 /(1+ np.exp(-x))

    def sigmoid_deriv(self,x):
        return x * (1-x)

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)
        #print("<<<<<<<", A, A[2].shape)

        error = A[-1] - y

        D = [error * self.sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A) - 2, 0, -1):
            #print("=======================", layer)
            #print("we", self.W[layer].shape, self.W[layer])
        
            #print("D", D[-1].shape)
            
            delta = D[-1].dot(self.W[layer].T)
            #print("delta1", delta.shape)
            delta = delta * self.sigmoid_deriv(A[layer])
            #print("delta2", delta.shape)
            D.append(delta)

        #print("<<<<<<<", D)

        D = D[::-1]
        #print("<<<<<<<", D)
        for layer in np.arange(0, len(self.W)):
            #print(layer)

            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def fit(self, X, y, epochs=1000, displayUpdate=1):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange (0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        
        return p
    
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss
    
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1], [0]])
#nn = NeuralNetwork([2, 2, 1], alpha=0.5)
#print(nn)
#nn.fit(X, y, epochs=5)
#
#
#for (x, target) in zip(X, y):
## make a prediction on the data point and display the result
## to our console
#    pred = nn.predict(x)[0][0]
#    step = 1 if pred > 0.5 else 0
#    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(    x, target[0], pred, step))