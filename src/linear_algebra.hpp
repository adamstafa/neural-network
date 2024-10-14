#pragma once

#include <vector>
#include <cassert>
#include <algorithm>
#include <random>

class matrix {
public:
    size_t rows;
    size_t cols;
    std::vector<double> data;

    matrix(size_t _rows, size_t _cols) : rows(_rows), cols(_cols), data(_rows * _cols) {}

    double& operator()(int row, int col)
    {
        assert(row >= 0 && row < rows && col >= 0 && col < cols);
        return data[row * cols + col];
    }

    const double& operator()(int row, int col) const
    {
        assert(row >= 0 && row < rows && col >= 0 && col < cols);
        return data[row * cols + col];
    }

    matrix operator*(const matrix& right)
    {
        // assert(right.rows == this->cols);

        auto& left = *this;
        auto res = matrix(left.rows, right.cols);
        for (size_t r = 0; r < left.rows; r++)
        {
            for (size_t c = 0; c < right.cols; c++)
            {
                double val = 0.0;
                for (size_t i = 0; i < left.cols; i++)
                {
                    val += left(r, i) * right(i, c);
                }
                res(r, c) = val;
            }
        }
        return res;
    }

    matrix transpose()
    {
        auto res = matrix(cols, rows);
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                res(c, r) = (*this)(r, c);
            }
        }
        return res;
    }

    matrix map(auto f)
    {
        auto res = matrix(rows, cols);
        std::transform(data.begin(), data.end(), res.data.begin(), f);
        return res;
    }

    matrix multiply_by(double num)
    {
        auto res = matrix(rows, cols);
        std::transform(data.begin(), data.end(), res.data.begin(), [num](double x) { return x * num; });
        return res;
    }

    matrix pointwise_multiply(const matrix& other)
    {
        auto res = matrix(std::min(rows, other.rows), std::min(cols, other.cols));
        for (size_t r = 0; r < res.rows; r++)
        {
            for (size_t c = 0; c < res.cols; c++)
            {
                res(r, c) = (*this)(r, c) * other(r, c);
            }
        }
        return res;
    }

    matrix operator-(const matrix& right)
    {
        auto res = matrix(std::min(rows, right.rows), std::min(cols, right.cols));
        for (size_t r = 0; r < res.rows; r++)
        {
            for (size_t c = 0; c < res.cols; c++)
            {
                res(r, c) = (*this)(r, c) - right(r, c);
            }
        }
        return res;
    }

    matrix operator+(const matrix& right)
    {
        auto res = matrix(std::min(rows, right.rows), std::min(cols, right.cols));
        for (size_t r = 0; r < res.rows; r++)
        {
            for (size_t c = 0; c < res.cols; c++)
            {
                res(r, c) = (*this)(r, c) + right(r, c);
            }
        }
        return res;
    }

    matrix L1_norm(double lambda)
    {
        auto res = matrix(rows, cols);
        std::transform(data.begin(), data.end(), res.data.begin(),
         [lambda](double x) { return (x > 0) ? lambda : -lambda; });
        return res;       
    }
};

matrix random_matrix(size_t rows, size_t cols, double deviation, long unsigned int seed = 42)
{
    std::mt19937 rng{seed};

    std::normal_distribution d{0.0, deviation};

    auto m = matrix(rows, cols);
    for (size_t i = 0; i < rows * cols; i++)
    {
        m.data[i] = d(rng);
    }
    return m;
}
