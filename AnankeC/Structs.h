#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <iostream>
#include <Eigen/Geometry>
#include <iostream>
#include <pagmo/problem.hpp>
#include <chrono>

using pagmo::vector_double;
namespace py = pybind11;

using ScaFuncType = std::function<double(Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;
using VecFuncType = std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;
using PrtFuncType = std::function<std::vector<Eigen::MatrixXd>(Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;
using LnkFuncType = std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;
using LprFuncType = std::function<std::vector<Eigen::MatrixXd>(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;

