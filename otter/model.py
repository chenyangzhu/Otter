from otter.history import history_recorder

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.compiled = False

    def compile(self, optimizer, loss, iteration, batch, gradient_method, minibatch):
        '''
        :param optimizer:           Optimizer 类
        :param loss:                Loss 类
        :param iteration:           # of iterations
        :param batch:               batch size, 如果是-1，则不进行 batch
        :param gradient_method:     gradient method = full / stochastic / minibatch
        :param minibatch:           mini-batch
        :return:
        '''

        self.optimizer = optimizer
        self.loss = loss
        self.iteration = iteration
        self.batch = batch
        self.gradient_method = gradient_method
        self.minibatch = minibatch
        self.compiled = True

    def record(self, recorder_list):
        self.recorder = history_recorder(recorder_list)

    def fit(self, X, y):
        # 首先看看有没有compile过
        if not self.compiled:
            raise AttributeError("You need to compile the model first.")

        self.X = X  # self.X 永远是最初的X，而 X 随着后面不断改变。
        self.y = y

        self.n = X.shape[0]

        # history
        hist_loss = []

        for i in range(self.iteration):

            if i % 10 == 0:
                print("The", i, "th iteration.")

            batch_loss = 0

            # Start batch
            for j in range(int(self.n / self.batch)):

                # Select Batch Data
                X = self.X[j * self.batch: (j+1) * self.batch]
                y = self.y[j * self.batch: (j+1) * self.batch]

                # 运行中间的layer
                for each_layer in self.layers:
                    X = each_layer.train_forward(X)

                # 运行最后的loss
                self.loss.train_forward(y=y, yhat=X)

                # Back-prop
                grad = self.loss.gradient  # Grad 是全局back-prop的 gradient
                for each_layer in reversed(self.layers):

                    each_layer.update_gradient(grad, method=self.gradient_method,
                                               minibatch=self.minibatch)
                    grad = each_layer.gradient['back']

                # Update Weights
                # 注意 learnable 与 gradient无关，只与update param有关
                for each_layer in self.layers:

                    if each_layer.learnable:  # 判断到底要不要update这层
                        self.optimizer.update_once(each_layer)

                batch_loss += self.loss.loss
            # End batch
            hist_loss.append(batch_loss / int(self.n / self.batch))
        # End iteration

        history = {'loss': hist_loss}
        return history

    def predict(self, X):
        for each_layer in self.layers:
            X = each_layer.train_forward(X)
        return X
