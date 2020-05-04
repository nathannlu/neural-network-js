const Neuron = require('./neuron');

// AND logic gate data set
const dataset = [
  { inputs: [0,0], outputs: [0] },
  { inputs: [0,1], outputs: [0] },
  { inputs: [1,0], outputs: [0] },
  { inputs: [1,1], outputs: [1] }
];

const inputs = [new Neuron(), new Neuron()]; // Input layer w/ 2 neurons
const hiddens = [new Neuron(), new Neuron()]; // Hidden layer w/ 2 neurons
const outputs = [new Neuron()] // Output layer w/ 1 neuron

// Connect input layer to hidden layer
inputs[0].connect(hiddens[0]);
inputs[0].connect(hiddens[1]);
inputs[1].connect(hiddens[0]);
inputs[1].connect(hiddens[1]);

// Connect hidden layer to output layer
hiddens[0].connect(outputs[0]);
hiddens[1].connect(outputs[0]);


// Passing information (through "brain")
const activate = input => {
  inputs.forEach((neuron, i) => neuron.activate(input[i]));
  hiddens.forEach(neuron => neuron.activate());
  return outputs.map(neuron => neuron.activate());
}

// Giving feedback (to "brain")
const propagate = target => {
  outputs.forEach((neuron, t) => neuron.propagate(target[t]));
  hiddens.forEach(neuron => neuron.propagate());
  return inputs.forEach(neuron => neuron.propagate());
}

// Looping
const train = (iterations = 1) => {
  while(iterations > 0) {
    dataset.map(datum => {
      activate(datum.inputs);
      propagate(datum.outputs);
    });
    iterations--;
  }
}

// Train network (10,000 iterations)
train(10000);

// Test network
console.log(activate([0,0]));
console.log(activate([0,1]));
console.log(activate([1,0]));
console.log(activate([1,1]));
