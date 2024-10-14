#pragma once

#include <vector>
#include <algorithm>
#include <iterator>
#include <map>
#include <functional>
#include <span>
#include <string>
#include <iostream>
#include <numeric>
#include <random>
#include <omp.h>
#include <ranges>
#include <limits>

#include "linear_algebra.hpp"

double activation_relu(double x)
{
    return std::max(0.0, x);
}

double activation_derivative_relu(double x)
{
    if (x <= 0)
        return 0;
    return 1;
}

double activation_id(double x)
{
    return x;
}

double activation_derivative_id(double x)
{
    return 1;
}

double activation_sigmoid(double x)
{
    return (1 / (1 + exp(-x)));
}

double activation_derivative_sigmoid(double x)
{
    return (exp(x) / std::pow(1 + exp(x), 2));
}

std::map<std::string, std::function<double (double)>> activation_map
    { {"relu", activation_relu}, {"id", activation_id}, {"sigmoid", activation_sigmoid} };
std::map<std::string, std::function<double (double)>> activation_derivative_map
    { {"relu", activation_derivative_relu}, {"id", activation_derivative_id}, {"sigmoid", activation_derivative_sigmoid} };

class layer
{
public:
    size_t size;
    matrix potential;
    matrix output;
    matrix activation_derivatives;
    matrix output_derivatives;
    matrix* weights;
    std::string activation_name;
    bool has_bias_neuron;

    layer(size_t _size, std::string _activation_name, bool _include_bias, matrix* _weights):
        size(_size),
        potential(matrix(_size + _include_bias, 1)),
        output(matrix(_size + _include_bias, 1)),
        activation_derivatives(matrix(_size + _include_bias, 1)),
        output_derivatives(_size, 1),
        weights(_weights),
        activation_name(_activation_name),
        has_bias_neuron(_include_bias)
    {
        if (_include_bias)
        {
            potential(size, 0) = 1;
            output(size, 0) = 1;
        }
    }

    void calculate_output(matrix& input)
    {
        auto new_potential = *weights * input;
        std::copy(new_potential.data.begin(), new_potential.data.end(), potential.data.begin());
        if (activation_name == "softmax")
        {
            auto exp_potentials = potential.map(exp);
            auto sum = std::reduce(exp_potentials.data.begin(), exp_potentials.data.end());
            output = exp_potentials.multiply_by(1.0 / sum);
        }
        else
        {
            output = potential.map(activation_map[activation_name]);
            activation_derivatives = potential.map(activation_derivative_map[activation_name]);
        }
    }
};

void add_weights(std::vector<matrix>& acc, std::vector<matrix>& in)
{
    for (size_t d = 0; d < acc.size(); d++)
    {
        for (size_t i = 0; i < acc[d].data.size(); i++)
        {
            acc[d].data[i] += in[d].data[i];
        }
    }
}

class mlp
{
public:
    std::vector<layer> layers;
    std::vector<matrix> weights;
    size_t depth;
    std::vector<matrix> old_weights;
    std::vector<std::vector<bool>> dropout_mask;
    std::mt19937 rng;

    double lambda_L2;
    double lambda_L1;
    double dropout_rate;

    mlp(std::vector<size_t> shape, double lL2 = 0.0, double lL1 = 0.0, double drop_rate = 0.0) :
        depth(shape.size() - 1), lambda_L2(lL2), lambda_L1(lL1), dropout_rate(drop_rate), rng(42)
    {
        for (size_t i = 0; i < depth; i++)
        {
            double dev = (i == 0) ? 1 / std::sqrt(shape[0]) : 1 / std::sqrt(shape[i] * 0.34);
            dev /= std::sqrt(1 - dropout_rate);
            // 0.34 ~ 0.5 - (1/sqrt(2*pi))^2
            weights.push_back(random_matrix(shape[i+1], shape[i] + 1, dev)); // +1 for fake bias
        }

        for (size_t i = 0; i <= depth; i++)
        {
            std::string activation_name = "relu";

            if (i == 0)
            {
                activation_name = "id";
            }
            else if (i == depth)
            {
                activation_name = "softmax";
            }

            layers.push_back(layer(shape[i], activation_name, i < depth, i == 0 ? nullptr : &weights[i - 1]));
        }
    }

    void eval(std::vector<double>& input)
    {
        assert(input.size() == layers[0].size);
        std::copy(input.begin(), input.end(), layers[0].output.data.begin());
        for (size_t i = 1; i <= depth; ++i)
        {
            layers[i].calculate_output(layers[i-1].output);
        }
    }

    int eval_category(std::vector<double>& input) {
        std::copy(input.begin(), input.end(), layers[0].output.data.begin());
        for (size_t i = 1; i <= depth; ++i)
        {
            for (size_t j = 0; j < layers[i-1].size - 1; j++)
            {
                layers[i-1].output(j, 0) = layers[i-1].output(j, 0) * (1 - dropout_rate);
            }
            layers[i].calculate_output(layers[i-1].output);
        }
        auto& outputs = layers[depth].output.data;
        auto max = std::max_element(outputs.begin(), outputs.end());
        return std::distance(outputs.begin(), max);
    }

    std::vector<matrix> gradient(std::vector<double>& input, int category)
    {
        eval(input);

        // backpropagation, calculate dE/dy
        // assuming softmax activation in output layer
        auto& output_layer = layers[depth];
        auto& out_der = output_layer.output_derivatives.data;
        std::fill(out_der.begin(), out_der.end(), 0);
        output_layer.output_derivatives(category, 0) = -1.0 / (output_layer.output(category, 0));
        auto exp_potentials = output_layer.potential.map(exp);
        auto sum = std::reduce(exp_potentials.data.begin(), exp_potentials.data.end());
        auto potential_derivatives = exp_potentials.multiply_by(- exp_potentials(category, 0) / std::pow(sum, 2));
        potential_derivatives(category, 0) = output_layer.output(category, 0) * (1 - output_layer.output(category, 0));
        potential_derivatives = potential_derivatives.multiply_by(output_layer.output_derivatives(category, 0));
        auto before_output_layer_derivatives = weights[depth-1].transpose() * potential_derivatives;
        std::copy(std::move_iterator(before_output_layer_derivatives.data.begin()), std::move_iterator(--before_output_layer_derivatives.data.end()),
                         layers[depth-1].output_derivatives.data.begin());

        for (int i = depth - 2; i >= 0; i--)
        {   
            // other layers
            auto& next_layer_derivatives = layers[i+1].output_derivatives;
            auto& activation_derivatives = layers[i+1].activation_derivatives;
            auto output_der = weights[i].transpose() * activation_derivatives.pointwise_multiply(next_layer_derivatives);
            std::copy(std::move_iterator(output_der.data.begin()), std::move_iterator(--output_der.data.end()),
                         layers[i].output_derivatives.data.begin());
        }

        // calculate dE/dw from dE/dy
        std::vector<matrix> gradients;
        for (size_t i = 0; i < depth - 1; i++)
        {
            gradients.push_back(layers[i+1].output_derivatives.pointwise_multiply(layers[i+1].activation_derivatives) * layers[i].output.transpose());
        }
        gradients.push_back(potential_derivatives * layers[depth-1].output.transpose());
        return gradients;
    }

    std::vector<matrix> zero_weights()
    {
        std::vector<matrix> result;
        for (auto& m: weights)
        {
            result.emplace_back(m.rows, m.cols);
        }
        return result;
    }

    std::vector<std::vector<bool>> generate_dropout_mask()
    {
        std::uniform_real_distribution<double> unif(0, 1);
        std::vector<std::vector<bool>> result;
        result.emplace_back(std::vector<bool>(layers[0].size, true));
        for (int i = 1; i < depth - 1; i++)
        {
            std::vector<bool> mask;
            for (int j = 0; j < layers[i].size - 1; j++)
            {
                mask.emplace_back(unif(rng) > dropout_rate);
            }
            mask.emplace_back(true);
            result.emplace_back(mask);
        }
        result.emplace_back(std::vector<bool>(layers[depth].size, true));
        return result;
    }

    std::vector<matrix> dropout_gradient(std::vector<std::pair<std::vector<double>, int>>& dataset, size_t batch_size, size_t offset)
    {
        dropout_mask = generate_dropout_mask();
        old_weights = weights;
        for (int d = 0; d < depth; d++)
        {
            for (int r = 0; r < weights[d].rows; r++)
            {
                if (!dropout_mask[d][r])
                {
                    for (int c = 0; c < weights[d].cols; c++)
                    {
                        weights[d](r, c) = 0;
                    }

                }
            }
        }
        auto grad = objective_function_gradient(dataset, batch_size, offset);
        weights = old_weights;
        return grad;
    }


    #pragma omp declare reduction(add_weights: std::vector<matrix>: add_weights(omp_out, omp_in))\
                            initializer (omp_priv=omp_orig)

    std::vector<matrix> objective_function_gradient(std::vector<std::pair<std::vector<double>, int>>& dataset, size_t batch_size, size_t offset)
    {
        auto result = zero_weights();
        size_t end = std::min(offset + batch_size, dataset.size());

        #pragma omp parallel num_threads(omp_get_num_procs()) reduction(add_weights: result)
        {
            auto mlp_copy = *this;

            #pragma omp for
            for (size_t i = offset; i < end; i++)
            {
                auto point = dataset[i];
                auto grad = mlp_copy.gradient(point.first, point.second);
                add_weights(result, grad);
            }
        }

        // elastic nets
        for (size_t d = 0; d < depth; d++)
        {
            result[d] = result[d] + weights[d].multiply_by(lambda_L2) + weights[d].L1_norm(lambda_L1);
        }

        return result;
    }

    void fit_adam(std::vector<std::pair<std::vector<double>, int>>& dataset, size_t epochs, size_t batch_size)
    {
        // https://browse.arxiv.org/pdf/1412.6980.pdf
        double alpha = 0.0012;
        double decay = 0.925;
        double beta_1 = 0.9;
        double beta_2 = 0.999;
        double epsilon = 0.00000001;
        auto m = zero_weights();
        auto m_hat = zero_weights();
        auto v = zero_weights();
        auto v_hat = zero_weights();
        auto g = zero_weights();
        double eps_stop = 0.1;
        
        double beta_1_t = 1;
        double beta_2_t = 1;

        size_t data_train_size = dataset.size() * 0.8;
        std::vector<std::pair<std::vector<double>, int>> data_train(dataset.begin(), dataset.begin() + data_train_size);
        std::vector<std::pair<std::vector<double>, int>> data_valid(dataset.begin() + data_train_size, dataset.end());

        // early stopping parameters
        double best_ce = std::numeric_limits<double>::infinity();
        std::mt19937 rng{42};
        std::vector<matrix> best_weights;
        size_t stop_after = 3;
        size_t no_improvement = 0;

        for (size_t i = 0; i < epochs; ++i) {
            alpha *= decay;
            auto ce = cross_entropy(data_valid);
            std::cout << "Training epoch " << i << "\n";
            std::cout << "Validation cross entropy: " << ce << "\n";

            if (ce > best_ce)
            {
                no_improvement++;
                if (no_improvement >= stop_after)
                {
                    std::cout << "Early stopping\n";
                    break;
                }
            }
            else
            {
                best_ce = ce;
                best_weights = weights;
                no_improvement = 0;
            }

            size_t offset = 0;
            size_t t = 0;
            std::shuffle(data_train.begin(), data_train.end(), rng);

            while (offset < data_train_size)
            {
                beta_1_t *= beta_1;
                beta_2_t *= beta_2;

                g = dropout_gradient(data_train, batch_size, offset);
                offset += batch_size;

                for (size_t d = 0; d < depth; d++)
                {
                    m[d] = m[d].multiply_by(beta_1) + (g[d].multiply_by(1 - beta_1));
                    v[d] = v[d].multiply_by(beta_2) + (g[d].pointwise_multiply(g[d]).multiply_by(1 - beta_2));
                }
                for (size_t d = 0; d < depth; d++)
                {
                    m_hat[d] = m[d].multiply_by(1/(1-beta_1_t));
                    v_hat[d] = v[d].multiply_by(1/(1-beta_2_t));
                }
                for (size_t d = 0; d < depth; d++)
                {
                    for (size_t i = 0; i < weights[d].data.size(); i++)
                    {
                        weights[d].data[i] = weights[d].data[i] - alpha * m_hat[d].data[i] / (std::sqrt(v_hat[d].data[i]) + epsilon);
                    }
                }
            }
        }
        
        weights = best_weights;
        auto ce = cross_entropy(data_valid);
        std::cout << "Final Validation cross entropy: " << ce << "\n";
    }

    double cross_entropy(std::vector<std::pair<std::vector<double>, int>>& dataset)
    {
        double result = 0.0;
        for (auto point : dataset)
        {            
            eval_category(point.first);
            result += -log(layers[depth].output(point.second, 0));
        }
        return result / dataset.size();
    }


    double accuracy(std::vector<std::pair<std::vector<double>, int>>& dataset)
    {
        double result = 0;
        for (auto point : dataset)
        {
            result += eval_category(point.first) == point.second;
        }
        return result / dataset.size();
    }
};
