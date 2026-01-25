# NeuraCommand

![neura_command](https://github.com/djeada/NeuraCommand/assets/37275728/bf778c51-e05f-4db4-9b36-1461462113dc)

NeuraCommand is a Python package and CLI tool designed for streamlined creation, training, and deployment of neural networks. It offers a clear-cut interface for managing complex neural architectures, including multi-layer perceptrons. The tool facilitates efficient data loading, model training, and application of models for predictive tasks. Aimed at providing practical functionality, NeuraCommand caters to both quick deployment needs and in-depth neural network experimentation.

## Features

- **Flexible Network Creation Library**: A library built on top of numpy, providing the flexibility to create various network configurations easily.

- **Intuitive Command-Line Interface (CLI)**: A user-friendly CLI that simplifies neural network operations, enhancing accessibility for usage in different programming languages or embedded systems.

- **Customizable Network Architectures**: Tailor your neural networks with a range of layers and structures. Create unique networks either through an interactive CLI experience or by populating a JSON template.

- **Matrix Input Support**: Offers compatibility with multiple matrix data formats, facilitating versatile network training. Once your network is set up, train it repeatedly using a variety of inputs.

- **Persistent Model Management**: Efficiently save and reload your neural network models for continuous training or future utilization. Once trained, models can be stored as blobs on disk and reloaded as needed.

- **Immediate Predictions**: Implement your trained networks for real-time prediction tasks. Input data can be manually entered or read from a disk, with the option to display predictions on-screen or save them to a file.

- **Layer Output Visualization**: Gain insights into your network's processing by displaying outputs from all layers. Choose between numerical printouts, CSV files, or other formats to keep track of your network's functionality—nothing remains hidden.

## Installation

*Note*: It's important to run these commands in the root directory of the repository for the installation to work correctly.

To install NeuraCommand, open your terminal or command prompt and run the following command:

```shell
pip install neura_command
```

After installing, you can verify the installation and check the installed version of NeuraCommand by running:

```shell
neura_command --version
```

This command should return the version number of NeuraCommand, indicating that it has been installed successfully.

## Usage of NeuraCommand CLI

The NeuraCommand CLI offers several actions: `create`, `train`, `predict`, `info`, `export`, `analyze`, and `visualize`. Each action is equipped with specific options to tailor its functionality.

```
python main.py --action {create,train,predict,info,export,analyze,visualize}
```

Options:

- `-h`, `--help`: Show the help message and exit.
- `--action {create,train,predict,info,export,analyze,visualize}`: Choose the action to perform.

### Create

Use the `create` action to create a new neural network. You need to provide a JSON file describing the network architecture and specify the output path for the created model.

```
python main.py --action create --network-architecture-json path/to/architecture.json --model-pkl path/to/output_model.pkl
```

Options:

- `--network-architecture-json`: Path to the JSON file that describes the neural network's architecture. This file should detail the layers, nodes, activation functions, etc.
- `--model-pkl`: [Optional] Destination path for saving the serialized neural network model (.pkl format). Defaults to 'network.pkl' if not specified.

### Train
Use the `train` action to train an existing network. You must provide the path to the network's `.pkl` file, input features, target values, number of epochs, and output model path (if not overwriting).

Example:

```
python main.py --action train --model-pkl path/to/network.pkl --features-csv path/to/features.csv --targets-csv path/to/targets.csv --epochs 100 --output-model-pkl path/to/output_model.pkl
```

Options:

- `--model-pkl`: Path to the neural network file (.pkl) to be trained.
- `--features-csv`: Path to the CSV file containing the input features for the training.
- `--targets-csv`: Path to the CSV file containing the target values for the training.
- `--epochs`: [Optional] Number of training epochs. Defaults to 100.
- `--overwrite`: [Optional] Flag to indicate if the existing .pkl file should be overwritten after training.
- `--output-model-pkl`: [Optional] Custom path to save the trained network model file (.pkl). Required if --overwrite is not set.
- `--save-history`: [Optional] Path to save training history as JSON file for later visualization.
- `--show-summary`: [Optional] Display training summary after training completes.

### Predict
Use the predict action to make predictions using a trained model. You need to provide the model `.pkl` file and input data. Additionally, specify how you want to output the predictions.

Example:

```
python main.py --action predict --model-pkl path/to/model.pkl --input-csv path/to/input.csv --output-mode display
```

To save predictions to a CSV file:

```
python main.py --action predict --model-pkl path/to/model.pkl --input-csv path/to/input.csv --output-mode save --output-csv path/to/output.csv
```

Options:

- `--model-pkl`: Path to the neural network file (.pkl) used for making predictions.
- `--input-csv`: Path to the input CSV file. The format of this file should match the input requirements of the model.
- `--output-mode`: Choose 'display' to show predictions on stdout or 'save' to save them to a CSV file.
- `--output-csv`: [Required if output-mode is 'save'] Path to save the prediction results as a CSV file.

### Analyze

Use the `analyze` action to perform detailed network analytics and diagnostics. This includes architecture analysis, weight statistics, performance metrics, and error analysis.

Example:

```
python main.py --action analyze --model path/to/model
```

To analyze with test data:

```
python main.py --action analyze --model path/to/model --features-csv path/to/features.csv --targets-csv path/to/targets.csv
```

Options:

- `--model`: Path to model (without extension).
- `--features-csv`: [Optional] Path to CSV file with input features for analysis.
- `--targets-csv`: [Optional] Path to CSV file with target values for error analysis.
- `--analysis-type`: Type of analysis to perform. Choices: `architecture`, `weights`, `performance`, `errors`, `all`. Default: `all`.
- `--format`: Output format. Choices: `text`, `json`. Default: `text`.
- `--task-type`: Task type for appropriate metrics. Choices: `regression`, `classification`. Default: `regression`.

The analyze command provides:
- **Architecture Analysis**: Layer-by-layer parameter counts, memory usage, optimizer, and loss function details.
- **Weight Statistics**: Per-layer and global statistics including mean, std, min/max values, and near-zero ratios.
- **Performance Metrics**: For regression (MSE, RMSE, MAE, R², MAPE) or classification (accuracy, precision, recall, F1, confusion matrix).
- **Error Analysis**: Error distribution, percentiles, worst predictions, and error breakdown by target range.

### Visualize

Use the `visualize` action to visualize training progress with ASCII charts in the terminal.

Example:

```
python main.py --action visualize --history-json path/to/training_history.json
```

Options:

- `--history-json`: Path to training history JSON file (saved from training with `--save-history`).
- `--metric`: Metric to visualize. Default: `loss`.
- `--width`: Width of ASCII chart. Default: 60.
- `--height`: Height of ASCII chart. Default: 20.
- `--format`: Output format. Choices: `ascii`, `json`, `table`. Default: `ascii`.

The visualize command provides:
- **ASCII Charts**: Terminal-friendly line charts showing training progress over epochs.
- **Table Output**: Epoch-by-epoch values in tabular format.
- **JSON Output**: Structured data with summary statistics for programmatic use.
- **Train/Val Gap Analysis**: Automatic overfitting detection when both training and validation losses are present.

## Quickstart for NeuraCommand package

Example code for using NeuraCommand in Python:

```python
import numpy as np

from neura_command.network.network_builder.string_network_builder import (
    StringNetworkBuilder,
)

# Example usage
builder = StringNetworkBuilder()
builder.add_layer(3, 5, "relu")  # First hidden layer
builder.add_layer(5, 3, "relu")  # Second hidden layer (output layer)

# Initialize the network using the builder
net = builder.build()

# Prepare the dataset for training
data_samples = [([0.1, 0.2, 0.3], [0, 1, 0])]  # Example: [(input_data, target_data)]
X = np.array([sample[0] for sample in data_samples])
Y = np.array([sample[1] for sample in data_samples])

# Train the network
net.train(X, Y, epochs=10, batch_size=1)

# Make predictions
test_input = np.array([[0.1, 0.2, 0.3]])
predicted_output = net.predict(test_input)
print(f"Predicted Output: {predicted_output}")

# Using the evaluate function
predicted_outputs = net.predict(X)
mse = np.mean((predicted_outputs - Y) ** 2)
print(f"Mean Squared Error: {mse}%")
```

## References

1. Nielsen, M. (2015). *Neural Networks and Deep Learning*. Determination Press.
   - [Online Book](http://neuralnetworksanddeeplearning.com/)

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
   - [MIT Press](https://www.deeplearningbook.org/)

3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
   - [Springer](http://www.springer.com/gp/book/9780387310732)

4. Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.
   - [Cambridge Core](https://www.cambridge.org/core/books/understanding-machine-learning/3059709FEBB4176FC37311D0AD8B2B8E)

5. Hertz, J., Krogh, A., & Palmer, R. G. (1991). *Introduction to the Theory of Neural Computation*. Addison-Wesley.
   - [Amazon](https://www.amazon.com/Introduction-Theory-Neural-Computation-Computational/dp/0201515601)

6. Fausett, L. (1994). *Fundamentals of Neural Networks: Architectures, Algorithms, and Applications*. Prentice-Hall.
   - [Pearson](https://www.pearson.com/us/higher-education/program/Fausett-Fundamentals-of-Neural-Networks-Architectures-Algorithms-and-Applications/PGM80736.html)

7. Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press.
   - [Amazon](https://www.amazon.com/Neural-Networks-Pattern-Recognition-Christopher/dp/0198538642)

8. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
   - [O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

9. Vasilev, I., & Slater, D. (2019). *Python Deep Learning*. Packt Publishing.
   - [Packt](https://www.packtpub.com/product/python-deep-learning-second-edition/9781789348460)

10. Iyengar, R. S. (2017). *Convolutional Neural Networks in Visual Computing: A Concise Guide*. CRC Press.
    - [Amazon](https://www.amazon.com/Convolutional-Networks-Visual-Computing-Concise/dp/1498749936)

## Contributing

We warmly welcome contributions to NeuraCommand! Whether you're fixing bugs, adding new features, improving documentation, or spreading the word, your help is invaluable.

### How to Contribute

1. **Fork the Repository**: Start by forking the NeuraCommand repository to your GitHub account.
2. **Create a Branch**: For each new feature or fix, create a new branch in your forked repository.
3. **Develop and Test**: Implement your changes and ensure they are thoroughly tested.
4. **Submit a Pull Request**: Once you're satisfied with your changes, submit a pull request to the main repository. Include a clear description of your changes and any relevant issue numbers.
5. **Code Review**: Your pull request will be reviewed by the maintainers. They may suggest changes or improvements.
6. **Merge**: Once your pull request is approved, it will be merged into the main codebase.

### Contribution Guidelines

- Ensure your code adheres to the project's coding standards and style.
- Update the documentation to reflect your changes, if necessary.
- Keep your commits clean and understandable.
- Stay respectful and open to feedback during the review process.

We're excited to see what you bring to NeuraCommand!

## License

NeuraCommand is licensed under the MIT License. This allows for a wide range of uses, providing flexibility and freedom for both personal and commercial applications while requiring only the retention of copyright and license notices. For more detailed information, please refer to the [LICENSE](LICENSE) file included in the repository.
