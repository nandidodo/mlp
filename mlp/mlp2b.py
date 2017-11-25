import numpy as np
import logging#
import sys
from mlp.data_providers import MNISTDataProvider

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit, SELUInit, UniformInit, NormalInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser



def train_model(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    return stats, keys, run_time


# Seed a random number generator
if len(sys.argv)>1:
    seed = int(sys.argv[1])
else:
    seed = 3209832

rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=batch_size, rng=rng)


#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 10, 100

#setup initialisations
relu_init = GlorotUniformInit(gain=0.5, rng=rng)
biases_init = ConstantInit(0.)




# selufanin = MultipleLayerModel([
#     AffineLayer(input_dim, hidden_dim, UniformInit(-np.sqrt(3/input_dim), np.sqrt(3/input_dim)), biases_init),
#     SELULayer(),
#     AffineLayer(hidden_dim, hidden_dim, UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim)), biases_init),
#     SELULayer(),
#     AffineLayer(hidden_dim, output_dim, UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim)), biases_init)
#     ])

# selufanout = MultipleLayerModel([
#     AffineLayer(input_dim, hidden_dim, UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim)), biases_init),
#     SELULayer(),
#     AffineLayer(hidden_dim, hidden_dim, UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim)), biases_init),
#     SELULayer(),
#     AffineLayer(hidden_dim, output_dim, UniformInit(-np.sqrt(3/output_dim), np.sqrt(3/output_dim)), biases_init)
#     ])



# layer_dic = {
#     "relu": ReluLayer(),
#     "elu": ELULayer(),
#     "selu": SELULayer(),
#     "leakyrelu": LeakyReluLayer()
# }

# layer = layer_dic[sys.argv[1]]  ##DELETE???????????
# init_list = [(arg, init_dic[arg]) for arg in sys.argv[2:]] ###DELETE??

def build_network(n_layers, activation, init_triple):
    input_init, hidden_init, output_init = init_triple
    l = [AffineLayer(input_dim, hidden_dim, input_init, biases_init), activation]
    for i in range(n_layers-1):
        l += [AffineLayer(hidden_dim, hidden_dim, hidden_init, biases_init), activation]
    l += [AffineLayer(hidden_dim, output_dim, output_init, biases_init)]

    return MultipleLayerModel(l)

error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)


for i in np.arange(10):
    seed += i
    print(seed)
    weights_rng = np.random.RandomState(seed)
    fanin_uniform = (
        UniformInit(-np.sqrt(3/input_dim), np.sqrt(3/input_dim), rng=weights_rng),
        UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim), rng=weights_rng),
        UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim), rng=weights_rng)
    )

    fanout_uniform = (
        UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim), rng=weights_rng),
        UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim), rng=weights_rng),
        UniformInit(-np.sqrt(3/output_dim), np.sqrt(3/output_dim), rng=weights_rng)
    )

    xavier_uniform = (
        UniformInit(-np.sqrt(6/(input_dim+hidden_dim)), np.sqrt(6/(input_dim+hidden_dim)), rng=weights_rng),
        UniformInit(-np.sqrt(3/hidden_dim), np.sqrt(3/hidden_dim), rng=weights_rng),
        UniformInit(-np.sqrt(6/(output_dim+hidden_dim)), np.sqrt(6/(output_dim+hidden_dim)), rng=weights_rng)
    )

    all_inits = [fanin_uniform, fanout_uniform, xavier_uniform]
    init_names = ["fanin_uniform", "fanout_uniform", "xavier_uniform"]
    init_dic = {name: triple for name, triple in zip(init_names, all_inits)}  ##DELETE??? 
    init_list = [("elu_" + name, triple) for (name, triple) in init_dic.items()]

    for name, triple in init_list:
        for n_layers in range(2,9):
            print(i, name, n_layers)
            m = build_network(n_layers, ELULayer(), triple)
            print(m)
            train_data.reset()
            valid_data.reset()
            stats, keys, run_time = train_model(m, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=False)
            path = '/afs/inf.ed.ac.uk/user/s17/s1786262/mlpractical/results2a/' + str(seed) + '_' + name + '_' + 'run' + str(i) + '_' + str(n_layers)
            np.save(path, stats)
