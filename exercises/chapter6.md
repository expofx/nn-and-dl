1. Backpropagation in a convolutional network The core equations of backpropagation in a network with fully-connected layers are (BP1)-(BP4) (link). Suppose we have a network containing a convolutional layer, a max-pooling layer, and a fully-connected output layer, as in the network discussed above. How are the equations of backpropagation modified?

The equation for computing the loss of the final fully connected layer is the same. We compute the error for one layer moving back for the max pooling (since it doesn't have any weights and biases to update). To update the right activation for the max pooling, we check if the activation for that neuron is the same as the max activation; if it is, we multiply 1 so that is updates, and if it isn't, it wasn't the value that was contributing to the error so we multiply 0. We modify the partial derivatives of b and w in the conv layers (kernel) to account for the shared weights and biases, where instead of the normal definition of a (sigmoid(wx+b)) we use a=sigmoid(w\*a_prev+b).

(A good link: https://datascience.stackexchange.com/questions/27506/back-propagation-in-cnn)

2. Using the tanh activation function Several times earlier in the book I've mentioned arguments that the tanh function may be a better activation function than the sigmoid function. We've never acted on those suggestions, since we were already making plenty of progress with the sigmoid. But now let's try some experiments with tanh as our activation function. Try training the network with tanh activations in the convolutional and fully-connected layers Begin with the same hyper-parameters as for the sigmoid network, but train for 20 epochs instead of 60. How well does your network perform? What if you continue out to 60 epochs? Try plotting the per-epoch validation accuracies for both tanh- and sigmoid-based networks, all the way out to 60 epochs. If your results are similar to mine, you'll find the tanh networks train a little faster, but the final accuracies are very similar. Can you explain why the tanh network might train faster? Can you get a similar training speed with the sigmoid, perhaps by changing the learning rate, or doing some rescaling\*\*You may perhaps find inspiration in recalling that σ(z)=(1+tanh(z/2))/2.? Try a half-dozen iterations on the learning hyper-parameters or network architecture, searching for ways that tanh may be superior to the sigmoid.

Tanh might train faster because it has a steeper derivative, e.g. weights and biases will update more per training step. Sigmoid could be modified by using a faster learning rate, or scaling the gradients slightly upwards.

Correction: Also, tanh can be negative so weights and biases will shift in different directions. You can subtract 1/2 from sigmoid to get a similar effect of having some negative values. However, final accuracies after many epochs are approximately equal.

3. The idea of convolutional layers is to behave in an invariant way across images. It may seem surprising, then, that our network can learn more when all we've done is translate the input data. Can you explain why this is actually quite reasonable?

It seems to be beneficial for two different reasons 1) more training data to accurately represent the data and 2) teaching it translational invariance on more of a global rather than local scale.

Correction: It wouldn't change the feature map that much because it will just be shifted by a single pixel, but the max pooling layers will be different. This in turn helps the FC layers learn better (since they don't have translational invariance inherently).

4. We've used the same initialization procedure for rectified linear units as for sigmoid (and tanh) neurons. Our argument for that initialization was specific to the sigmoid function. Consider a network made entirely of rectified linear units (including outputs). Show that rescaling all the weights in the network by a constant factor c>0 simply rescales the outputs by a factor cL−1, where L is the number of layers. How does this change if the final layer is a softmax? What do you think of using the sigmoid initialization procedure for the rectified linear units? Can you think of a better initialization procedure? Note: This is a very open-ended problem, not something with a simple self-contained answer. Still, considering the problem will help you better understand networks containing rectified linear units.

relu(wa+b) means that each time, if c>0 the a will be c\*a larger than without the c factor. Since the feedforward part of the network continually takes the previous layer activations as inputs and does another computation, the c will be multiplied L-1 times. If the final layer is a softmax, this likely means the final output will be very small if c>1 (the right tail end of the softmax) or close to 0.5 if c<1. A better initialization procedure may be to initialize with smaller weights.

Correction: Another way to explain c^(L-1) is that ReLU(cz)=c\*ReLU(z). With softmax, the final output classification is still the same, but if c>1 it is more confident in its classification and if c<1 it is less confident.

5. Our analysis of the unstable gradient problem was for sigmoid neurons. How does the analysis change for networks made up of rectified linear units? Can you think of a good way of modifying such a network so it doesn't suffer from the unstable gradient problem?

There isn't as much of a problem of vanishing gradients since more multiplying by the derivative of ReLU will pretty much keep it constant. However, there is a risk that some neurons may not learn as effectively since if it is negative it will be set to 0.

Correction: To fix the "dying ReLU" problem, use a leaky ReLU or ELU.
