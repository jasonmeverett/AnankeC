#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Geometry>
#include <iostream>
#include "Dynamics.h"

namespace py = pybind11;


struct Ex1
{
	static Eigen::VectorXd f(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::VectorXd dX(2);
		double mult = py::cast<double>(params[0]);
		dX[0] = X[1];
		dX[1] = U[0] * mult;
		return dX;
	};

	static Eigen::MatrixXd dfdX(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::MatrixXd dX = Eigen::MatrixXd::Zero(2,2);
		dX(0, 1) = 1.0;
		return dX;
	};

	static Eigen::MatrixXd dfdU(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::MatrixXd dX = Eigen::MatrixXd::Zero(2, 1);
		double mult = py::cast<double>(params[0]);
		dX(1, 0) = mult;
		return dX;
	};

	static double Jctrl(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		return U[0] * U[0];
	};

	static Eigen::MatrixXd dJctrl(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::MatrixXd dJ = Eigen::MatrixXd::Zero(1, 3);
		dJ(0, 2) = 2.0 * U[0];
		return dJ;
	};

	static Eigen::VectorXd g1(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::VectorXd g(2);
		g[0] = X[0];
		g[1] = X[1];
		return g;
	};

	static Eigen::MatrixXd dg1(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::MatrixXd dg = Eigen::MatrixXd::Zero(2, 3);
		dg(0, 0) = 1.0;
		dg(1, 1) = 1.0;
		return dg;
	};

	static Eigen::VectorXd g2(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::VectorXd g(2);
		g[0] = X[0] - 1.0;
		g[1] = X[1];
		return g;
	};

	static Eigen::MatrixXd dg2(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::MatrixXd dg = Eigen::MatrixXd::Zero(2, 3);
		dg(0, 0) = 1.0;
		dg(1, 1) = 1.0;
		return dg;
	};

	static Eigen::VectorXd g3(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::VectorXd g(1);
		double ulim = py::cast<double>(params[0]);
		g[0] = sqrt(U[0] * U[0]) - ulim;
		return g;
	};

	static Eigen::MatrixXd dg3(Eigen::VectorXd X, Eigen::VectorXd U, double T, py::list params) {
		Eigen::MatrixXd dg = Eigen::MatrixXd::Zero(1, 3);
		dg(0, 2) = U[0] / sqrt(U[0] * U[0]);
		return dg;
	};

	static void Build(py::module& m)
	{
		auto obj = py::class_<Ex1>(m, "ex1");
		obj.def_static("f", &Ex1::f);
		obj.def_static("dfdX", &Ex1::dfdX);
		obj.def_static("dfdU", &Ex1::dfdU);
		obj.def_static("Jctrl", &Ex1::Jctrl);
		obj.def_static("dJctrl", &Ex1::dJctrl);
		obj.def_static("g1", &Ex1::g1);
		obj.def_static("g2", &Ex1::g2);
		obj.def_static("g3", &Ex1::g3);
		obj.def_static("dg1", &Ex1::dg1);
		obj.def_static("dg2", &Ex1::dg2);
		obj.def_static("dg3", &Ex1::dg3);
	}
};
