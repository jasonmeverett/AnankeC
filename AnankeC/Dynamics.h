#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Geometry>
#include <iostream>
#include <pagmo/problem.hpp>

using pagmo::vector_double;
namespace py = pybind11;

using ScaFuncType = std::function<double(							Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;
using VecFuncType = std::function<Eigen::VectorXd(					Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;
using MatFuncType = std::function<Eigen::MatrixXd(					Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;
using PrtFuncType = std::function<std::vector<Eigen::MatrixXd>(		Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;

using LnkFuncType = std::function<Eigen::VectorXd				(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;
using LprFuncType = std::function<std::vector<Eigen::MatrixXd>	(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double, vector_double)>;