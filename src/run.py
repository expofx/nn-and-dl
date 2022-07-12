### chapter 1

import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data=list(training_data)
validation_data=list(validation_data)

# net = network.Network([784, 100, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# exercise
# net = network.Network([784, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

### chapter 2

# import network_matrix

# net = network_matrix.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

### chapter 3

# normal

# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.SGD(training_data, 30, 10, 0.5,
#         lmbda=5.0,
#         evaluation_data=validation_data,
#         monitor_evaluation_accuracy=True,
#         monitor_evaluation_cost=False,
#         monitor_training_accuracy=False,
#         monitor_training_cost=False
# )

# l1 regularization

# import network2_l1
# net = network2_l1.Network([784, 30, 10], cost=network2_l1.CrossEntropyCost)
# net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

# early stopping

# import network2_early
# net = network2_early.Network([784, 30, 10], cost=network2_early.CrossEntropyCost)
# net.SGD(training_data[:500], 30, 10, 0.5,
#         lmbda=5.0,
#         evaluation_data=validation_data[:100],
#         monitor_evaluation_accuracy=True,
#         monitor_evaluation_cost=True,
#         monitor_training_accuracy=True,
#         monitor_training_cost=True,
#         early_stopping=True,
#         early_stopping_n=4
# )

# learning rate

# import network2_lr
# net = network2_lr.Network([784, 30, 10], cost=network2_lr.CrossEntropyCost)
# net.SGD(training_data[:500], 30, 10, eta=1.0,
#         lmbda=5.0,
#         evaluation_data=validation_data[:100],
#         monitor_evaluation_accuracy=True,
#         monitor_evaluation_cost=True,
#         monitor_training_accuracy=True,
#         monitor_training_cost=True,
#         early_stopping=True,
#         early_stopping_n=10
# )

# momentum

# import network2_momentum
# net = network2_momentum.Network([784, 30, 10], cost=network2_momentum.CrossEntropyCost)
# net.SGD(training_data[:500], 30, 10, eta=1.0,
#         lmbda=5.0,
#         evaluation_data=validation_data[:100],
#         monitor_evaluation_accuracy=True,
#         monitor_evaluation_cost=True,
#         monitor_training_accuracy=True,
#         monitor_training_cost=True,
#         mu=0.5
# )

### chapter 6

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

net = network3.Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20*12*12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)