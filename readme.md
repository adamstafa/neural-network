# Neural network
A neural network for classifying the images from the Fashion-MNIST dataset with over 90% accuracy. Written from scratch using C++ without any ML libraries.

## Features
- Customizable feed-forward architecture
- Training using mini-batch gradient descent and backpropagation
- Softmax output layer and categorical cross entropy loss
- Glorot weight initialization
- Input standardization
- Dropout and L1 and L2 regularization for better generalization
- Adam optimizer
- Early stopping based on the loss on the validation set
- Parallelization using OpenMP
- Final accuracy over 90%

## Usage
### Build the project
```bash
mkdir build
cd build
cmake ..
make
```
### Train the model
You can run the training using the `network` binary. The network learns the weights from the train data and predicts the labels of the test data.
```bash
./network ../data  # provide path to the directory containing the data sets
```
### Evaluate the accuracy
You can check the accuracy by using the included evaluator.
```bash
python3 ../evaluator/evaluate.py test_predictions.csv ../data/fashion_mnist_test_labels.csv
```
