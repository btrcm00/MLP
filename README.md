## Neural Network

### Activation

#### Sigmoid

- Defining:
<div align="center"><img src="img\sigmoid.jpg"></div>

- Derivative:
<div align="center"><img src="img\derivative-sigmoid.jpg"></div>

- Cons:
  - Not a zero centric function.
  - Suffers with gradient vanishing.
  - Output of values which are far away from centroid is close to zero.
  - Computationally expensive because it has to calculate exponential value in function.
- Cons:
  - Smooth gradient, preventing “jumps” in output values.
  - Output values bound between 0 and 1, normalizing the output of each neuron.

#### Tanh

- Defining:
<div align="center"><img src="img\tanh.jpg"></div>

- Derivative:
<div align="center"><img src="img\derivative-tanh.jpg"></div>

- Cons:
  - more computation expensive than sigmoid function.
  - suffers with gradient vanishing.
  - output of values which are far away from centroid is close to zero.
- Pros:
  - have all advantages of sigmoid function and it also a zero centric function.

#### ReLU

- Defining:
<div align="center"><img src="img\ReLU.jpg"></div>

- Derivative:
<div align="center"><img src="img\derivative-ReLU.jpg"></div>

- Cons:
  - No matter what for negative values neuron is completely inactive.
  - Non zero centric function.
- Pros:
  - No gradient vanishing
  - Derivative is constant
  - Less computation expensive

#### PReLU

- Defining:
<div align="center"><img src="img\prelu.jpg"></div>

- Derivative:
<div align="center"><img src="img\derivative-prelu.jpg"></div>

- Cons:
  - it can’t be used for the complex Classification. It lags behind the Sigmoid and Tanh for some of the use cases.
- Pros:
  - are one attempt to fix the “dying ReLU” problem by having a small negative slope (of 0.01, or so).

#### ELU

- Defining:
<div align="center"><img src="img\elu.jpg"></div>

- Derivative:
<div align="center"><img src="img\derivative-elu.jpg"></div>

- Cons: For x > 0, it can blow up the activation with the output range of [0, inf].
- Pros:
  - help alleviate the vanishing gradient
  - it has a nonzero gradient when x < 0, avoids the dying units issues.
  - is smooth every where, helps speed up GD

#### Speed: elu > prelu > relu > tanh > logistic

#### Softmax

- Defining:
<div align="center"><img src="img\softmax.jpg"></div>

- Derivative:
<div align="center"><img src="img\derivative-softmax.jpg"></div>

<p align="center">
<img src="img/softmax1.png">
</p>

### Run

- `python mlp.py`
