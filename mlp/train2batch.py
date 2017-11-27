#re-initialize neural network weights and reset the data providers so you get a properly initialized experiment.

#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.style.use('ggplot')

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
    
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(patience=10, max_num_epochs=num_epochs, stats_interval=stats_interval)

    # # Plot the change in the validation and training set error over training.
    # fig_1 = plt.figure(figsize=(8, 4))
    # ax_1 = fig_1.add_subplot(111)
    # for k in ['error(train)', 'error(valid)']:
    #     ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
    #               stats[1:, keys[k]], label=k)
    # ax_1.legend(loc=0)
    # ax_1.set_xlabel('Epoch number')

    # # Plot the change in the validation and training set accuracy over training.
    # fig_2 = plt.figure(figsize=(8, 4))
    # ax_2 = fig_2.add_subplot(111)
    # for k in ['acc(train)', 'acc(valid)']:
    #     ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
    #               stats[1:, keys[k]], label=k)
    # ax_2.legend(loc=0)
    # ax_2.set_xlabel('Epoch number')
    
    return stats, keys, #run_time, fig_1, ax_1, fig_2, ax_2



    # The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)




# The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer, DropoutLayer, BatchNormalizationLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, Adam, RMSProp
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim = 784, 47




def build_network(hidden_dim, n_layers, activation, incl_prob=None, batchnm=True):
    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)
    if batchnm:
        act = [BatchNormalizationLayer(hidden_dim), activation]
        act_in = [BatchNormalizationLayer(input_dim), activation]
    else:
        act = [activation]
        act_in = [activation]

    l = [AffineLayer(input_dim, hidden_dim, weights_init, biases_init)] + act_in
    for i in range(n_layers-1):
        if incl_prob is not None:
            l += [DropoutLayer(rng=rng, incl_prob=incl_prob), \
            AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init)] + act
        else:
            l += [AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init)] + act
        
    l += [AffineLayer(hidden_dim, output_dim, weights_init, biases_init)]

    return MultipleLayerModel(l)


error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rules = Adam, RMSProp, GrdientDescentLearningRule
lrule_names = ['adam', 'rms', 'gds']

#Remember to use notebook=False when you write a script to be run in a terminal
for i in np.arange(20):
    seed += i
    for name, learning_rule in zip(lrule_names, learning_rules):
        for l_rate in [0.001, 0.01, 0.1]:
            if l_rate == 0.1 and name != 'sgd':
                continue
            model = build_network(300, 4, ReluLayer(), incl_prob=0.8)
            train_data.reset()
            valid_data.reset()
            stats = train_model_and_plot_stats(
                model, error, learning_rule(learning_rate=l_rate), train_data, valid_data, num_epochs, stats_interval, notebook=False)
            path = '/afs/inf.ed.ac.uk/user/s17/s1786262/mlpractical/results_batchnorm/' + str(seed) + '_' + 'run' + str(i) + '_' + str(name) + '_' + str(l_rate)
            np.save(path, stats)