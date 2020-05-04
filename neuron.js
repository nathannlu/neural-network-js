const uid = require("cuid");

function Neuron(bias) {
  this.id = uid();

  // this.bias ∈ ℝ && -1 < this.bias < 1
  this.bias = bias == undefined ? Math.random() * 2 - 1 : bias;

  // Neuron connections
  this.incoming = {
    targets: {},
    weights: {},
  };
  this.outgoing = {
    targets: {},
    weights: {},
  };

  this._output; // f'(x)
  this.output; // f(x)
  this.error; // E(f(x))
  this._error; // E'(f(x))

  // Define what neuron(s) a particular neuron
  // has to "report to"
  // Connects this neuron to the next
  this.connect = (neuron, weight) => {
    this.outgoing.targets[neuron.id] = neuron;
    neuron.incoming.targets[this.id] = this;
    this.outgoing.weights[neuron.id] = neuron.incoming.weights[this.id] =
      weight == undefined ? Math.random() * 2 - 1 : weight; // weight ∈ ℝ && -1 < weight < 1
  };

  // Activation function
  this.activate = (input) => {
    const self = this;

    // f(x) = 1 / (1 + e^(-x))
    const sigmoid = (x) => {
      return 1 / (1 + Math.exp(-x));
    };
    const _sigmoid = (x) => {
      return sigmoid(x) * (1 - sigmoid(x));
    };

    // Input neurons
    if (input != undefined) {
      this._output = 1; // f'(x)
      this.output = input; // f(x)
    }

    // Hidden/output neurons
    else {
      const sum = Object.keys(this.incoming.targets).reduce(
        (total, target, index) => {
          return (total +=
            self.incoming.targets[target].output *
            self.incoming.weights[target]);
        },
        this.bias
      );

      this._output = _sigmoid(sum); // f'(x)
      this.output = sigmoid(sum); // f(x)
    }

    return this.output;
  };

  this.propagate = (target, rate = 0.3) => {
    const self = this;

    //𝛿E /𝛿squash
    const sum = target == undefined ? Object.keys(this.outgoing.targets).reduce(function(total, target, index) {
      // Δweight
      self.outgoing.targets[target].incoming.weights[self.id] = self.outgoing.weights[target] -= rate * self.outgoing.targets[target].error * self.output;
      
      return total += self.outgoing.targets[target].error * self.outgoing.weights[target];
    }, 0) : this.output - target;

    // 𝛿squash/𝛿sum
    this.error = sum * this._output
    
    // Δbias
    this.bias -= rate * this.error;
    
    return this.error;
  }
};

module.exports = Neuron;
