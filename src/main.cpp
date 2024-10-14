#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

#include "network.hpp"

class trainer
{
    public:
    mlp net;
    double mean;
    double deviation;
    std::vector<std::pair<std::vector<double>, int>> train_set;

    trainer(std::vector<size_t> layer_sizes, double lambdaL2, double lambdaL1, double dropout_rate):
        net(layer_sizes, lambdaL2, lambdaL1, dropout_rate), mean(), deviation(), train_set() {}


    void calculate_normalization_params(std::vector<std::pair<std::vector<double>, int>>& dataset)
    {
        double sum = 0;
        size_t count = 0;
        for (auto &point : dataset)
        {
            for (auto val : point.first)
            {
                sum += val;
                ++count;
            }
        }
        mean = sum / count;
        double variance_sum = 0;
        for (auto &point : dataset)
        {
            for (auto val : point.first)
            {
                variance_sum += std::pow((val - mean), 2);
            }
        }
        deviation = std::sqrt(variance_sum / (count - 1));
    }

    void normalize_vector(std::vector<double> &vector)
    {
        for (auto &val : vector)
        {
            val = (val - mean) / deviation;
        }
    }

    void normalize_dataset(std::vector<std::pair<std::vector<double>, int>>& dataset)
    {
        for (auto &point : dataset)
        {
            normalize_vector(point.first);
        }
    }

    void load_train_set(std::string features_path, std::string labels_path)
    {
        std::ifstream train_vectors(features_path);
        std::ifstream train_labels(labels_path);
        std::string line;
        while (std::getline(train_vectors, line))
        {
            int output;
            train_labels >> output;
            std::vector<double> input_vector;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ','))
            {
                double value = std::stod(cell);
                input_vector.push_back(value);
            }
            train_set.push_back({ input_vector, output });
        }
        calculate_normalization_params(train_set);
        normalize_dataset(train_set);
    }

    void fit(int epochs, int batch_size)
    {
        net.fit_adam(train_set, epochs, batch_size);
    }

    void label_test_set(std::string features_path, std::string output_path)
    {
        std::ifstream test_vectors(features_path);
        std::ofstream predictions(output_path);
        std::string line;
        while (std::getline(test_vectors, line))
        {
            std::vector<double> input_vector;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ','))
            {
                double value = std::stod(cell);
                input_vector.push_back(value);
            }
            normalize_vector(input_vector);
            predictions << net.eval_category(input_vector) << std::endl;
        }
    }
};

int main(int argc, char* argv[])
{
    // network hyperparameters
    std::vector<size_t> layer_sizes {28*28, 420, 210, 69, 10};
    double lambdaL2 = 0.02;
    double lambdaL1 = 0.001;
    double dropout_rate = 0.13;
    int epochs = 20;
    int batch_size = 128;
    
    // configure paths to the dataset
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path to data>" << std::endl;
        return 1;
    }
    std::filesystem::path data_path { argv[1] };
    auto train_features = data_path / "fashion_mnist_train_vectors.csv";
    auto train_labels = data_path / "fashion_mnist_train_labels.csv";
    auto test_features = data_path / "fashion_mnist_test_vectors.csv";
    auto output = "./test_predictions.csv";


    auto fashion_mnist_trainer = trainer(layer_sizes, lambdaL2, lambdaL1, dropout_rate);
    fashion_mnist_trainer.load_train_set(train_features, train_labels);
    fashion_mnist_trainer.fit(epochs, batch_size);
    fashion_mnist_trainer.label_test_set(test_features, output);

    return 0;
}
