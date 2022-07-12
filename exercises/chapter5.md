1. In our discussion of the vanishing gradient problem, we made use of the fact that |σ′(z)|<1/4. Suppose we used a different activation function, one whose derivative could be much larger. Would that help us avoid the unstable gradient problem?

Yes, a ReLU function could help avoid the unstable gradient problem since the derivative of z>0 is 1, which wouldn't shrink the gradient each layer back but rather keep it relatively constant (at least in terms of the activation function and assuming the weights aren't too large or small).

2. Consider the product |wσ′(wa+b)|. Suppose |wσ′(wa+b)|≥1. (1) Argue that this can only ever occur if |w|≥4. (2) Supposing that |w|≥4, consider the set of input activations a for which |wσ′(wa+b)|≥1. Show that the set of a satisfying that constraint can range over an interval no greater in width than
   2|w|ln(|w|(1+1−4/|w|−−−−−−−−√)2−1).(123)
   (3) Show numerically that the above expression bounding the width of the range is greatest at |w|≈6.9, where it takes a value ≈0.45. And so even given that everything lines up just perfectly, we still have a fairly narrow range of input activations which can avoid the vanishing gradient problem.

(1) Since σ′ has a max of 1/4, |w| has to be greater than or equal to 4.
(2) Plug in variables and solve for a. Correction: assume that b is 0 since σ′ has a max at 0 so this will give the largest range.
(3) By plotting the function, we see that it takes on a max at |w|~6.9.

3. Identity neuron: Consider a neuron with a single input, x, a corresponding weight, w1, a bias b, and a weight w2 on the output. Show that by choosing the weights and bias appropriately, we can ensure w2σ(w1x+b)≈x for x∈[0,1]. Such a neuron can thus be used as a kind of identity neuron, that is, a neuron whose output is the same (up to rescaling by a weight factor) as its input. Hint: It helps to rewrite x=1/2+Δ, to assume w1 is small, and to use a Taylor series expansion in w1Δ.

As a rough heuristic, the sigmoid(z) will always be between 0 and 1, and you can set w2 to the value proportional to 1/sigmoid to get back x.
