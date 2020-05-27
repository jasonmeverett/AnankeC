#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Geometry>
#include <iostream>

namespace py = pybind11;

using ScaFuncType = std::function<double(Eigen::VectorXd, Eigen::VectorXd, double, py::list)>;
using VecFuncType = std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd, double, py::list)>;
using JacFuncType = std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd, double, py::list)>;
using LnkFuncType = std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double, py::list)>;
