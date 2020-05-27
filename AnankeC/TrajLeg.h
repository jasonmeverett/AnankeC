#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Geometry>
#include <iostream>
#include "Dynamics.h"

enum RegionFlags
{
	FRONT = 1,
	BACK = 2,
	PATH = 3
};

enum ObjectiveFlags
{
	LAGRANGE = 1,
	MAYER = 2,
	BOLZA = 3,
	TIME = 4
};

struct TrajLeg
{
	py::tuple f;
	py::tuple J;
	py::list dynParams;
	py::list coneqs_f;
	py::list coneqs_b;
	py::list coneqs_p;
	py::list conins_f;
	py::list conins_b;
	py::list conins_p;

	double T;
	int num_nodes;
	int lenX;
	int lenU;
	int lenN;
	bool dynamicsSet;
	bool objSet;
	ObjectiveFlags objType;
	double Tmin;
	double Tmax;
	bool TOFset;
	py::list bnds_min;
	py::list bnds_max;

	TrajLeg(int n, double T){
		this->num_nodes = n;
		this->T = T;
	};

	void set_dynamics(VecFuncType f, JacFuncType dfdX, JacFuncType dfdU, py::list params)
	{
		this->f = py::make_tuple(f, dfdX, dfdU, params);
		this->dynamicsSet = true;
	}

	Eigen::VectorXd testDynamics(Eigen::VectorXd X, Eigen::VectorXd T, double U)
	{
		Eigen::VectorXd fx = py::cast<VecFuncType>(this->f[0])(X, T, U, this->f[3]);
		return fx;
	}

	void set_len_X_U(int X, int U)
	{
		this->lenX = X;
		this->lenU = U;
		this->lenN = X + U;
	}

	void add_eq(VecFuncType con, JacFuncType dcon, int lcon, RegionFlags reg, py::list params, bool td)
	{
		switch (reg)
		{
		case FRONT:
			coneqs_f.append(py::make_tuple(con, dcon, lcon, params, td));
			break;
		case BACK:
			coneqs_b.append(py::make_tuple(con, dcon, lcon, params, td));
			break;
		case PATH:
			coneqs_p.append(py::make_tuple(con, dcon, lcon, params, td));
			break;
		}
		return;
	}

	void add_ineq(VecFuncType con, JacFuncType dcon, int lcon, RegionFlags reg, py::list params, bool td)
	{
		switch (reg)
		{
		case FRONT:
			conins_f.append(py::make_tuple(con, dcon, lcon, params, td));
			break;
		case BACK:
			conins_b.append(py::make_tuple(con, dcon, lcon, params, td));
			break;
		case PATH:
			conins_p.append(py::make_tuple(con, dcon, lcon, params, td));
			break;
		}
		return;
	}

	int getTotLength()
	{
		int totLen = 1 + this->num_nodes * (this->lenX + this->lenU);
		return totLen;
	}

	void set_TOF(double Tmin, double Tmax)
	{
		this->Tmin = Tmin;
		this->Tmax = Tmax;
		this->TOFset = true;
	}

	void set_bounds(py::list bnds_min, py::list bnds_max)
	{
		this->bnds_min = bnds_min;
		this->bnds_max = bnds_max;
	}

	void set_obj(ScaFuncType f, JacFuncType df, ObjectiveFlags typ, py::list params)
	{
		this->J = py::make_tuple(f, df, params);
		this->objType = typ;
		this->objSet = true;
	}


	static void Build(py::module& m)
	{
		py::enum_<RegionFlags>(m, "RegionFlags")
			.value("FRONT", RegionFlags::FRONT)
			.value("BACK", RegionFlags::BACK)
			.value("PATH", RegionFlags::PATH)
			.export_values();
		py::enum_<ObjectiveFlags>(m, "ObjectiveFlags")
			.value("LAGRANGE", ObjectiveFlags::LAGRANGE)
			.value("MAYER", ObjectiveFlags::MAYER)
			.value("BOLZA", ObjectiveFlags::BOLZA)
			.value("TIME", ObjectiveFlags::TIME)
			.export_values();

		auto obj = py::class_<TrajLeg>(m, "TrajLeg");
		obj.def(py::init<int, double>());
		obj.def("set_dynamics", &TrajLeg::set_dynamics);
		obj.def("testDynamics", &TrajLeg::testDynamics);
		obj.def("set_len_X_U", &TrajLeg::set_len_X_U);
		obj.def("add_eq", &TrajLeg::add_eq);
		obj.def("add_ineq", &TrajLeg::add_ineq);
		obj.def("getTotLength", &TrajLeg::getTotLength);
		obj.def("set_TOF", &TrajLeg::set_TOF);
		obj.def("set_bounds", &TrajLeg::set_bounds);
		obj.def("set_obj", &TrajLeg::set_obj);
		
		obj.def_readwrite("num_nodes", &TrajLeg::num_nodes);
		obj.def_readwrite("T", &TrajLeg::T);
		obj.def_readwrite("f", &TrajLeg::f);
		obj.def_readwrite("J", &TrajLeg::J);
		obj.def_readwrite("dynamicsSet", &TrajLeg::dynamicsSet);
		obj.def_readwrite("coneqs_f", &TrajLeg::coneqs_f);
		obj.def_readwrite("coneqs_b", &TrajLeg::coneqs_b);
		obj.def_readwrite("coneqs_p", &TrajLeg::coneqs_p);
		obj.def_readwrite("conins_f", &TrajLeg::conins_f);
		obj.def_readwrite("conins_b", &TrajLeg::conins_b);
		obj.def_readwrite("conins_p", &TrajLeg::conins_p);
		obj.def_readwrite("lenX", &TrajLeg::lenX);
		obj.def_readwrite("lenU", &TrajLeg::lenU);
		obj.def_readwrite("lenN", &TrajLeg::lenN);
		obj.def_readwrite("objSet", &TrajLeg::objSet);
		obj.def_readwrite("objType", &TrajLeg::objType);
		obj.def_readwrite("Tmin", &TrajLeg::Tmin);
		obj.def_readwrite("Tmax", &TrajLeg::Tmax);
		obj.def_readwrite("TOFset", &TrajLeg::TOFset);
		obj.def_readwrite("bnds_min", &TrajLeg::bnds_min);
		obj.def_readwrite("bnds_max", &TrajLeg::bnds_max);

		obj.def(
			py::pickle(
				[](const TrajLeg& tl) {
					return py::make_tuple(
						tl.num_nodes, 
						tl.T,
						tl.f,
						tl.J,
						tl.dynamicsSet,
						tl.coneqs_f,
						tl.coneqs_b,
						tl.coneqs_p,
						tl.conins_f,
						tl.conins_b,
						tl.conins_p,
						tl.lenX,
						tl.lenU,
						tl.lenN,
						tl.objSet,
						tl.objType,
						tl.Tmin,
						tl.Tmax,
						tl.TOFset,
						tl.bnds_min,
						tl.bnds_max
						);
				},
				[](py::tuple t) {
					TrajLeg tl(t[0].cast<int>(), t[1].cast<double>());
					tl.num_nodes = t[0].cast<int>();
					tl.T = t[1].cast<double>();
					tl.f = t[2].cast<py::tuple>();
					tl.J = t[3].cast<py::tuple>();
					tl.dynamicsSet = t[4].cast<bool>();
					tl.coneqs_f = t[5].cast<py::list>();
					tl.coneqs_b = t[6].cast<py::list>();
					tl.coneqs_p = t[7].cast<py::list>();
					tl.conins_f = t[8].cast<py::list>();
					tl.conins_b = t[9].cast<py::list>();
					tl.conins_p = t[10].cast<py::list>();
					tl.lenX = t[11].cast<int>();
					tl.lenU = t[12].cast<int>();
					tl.lenN = t[13].cast<int>();
					tl.objSet = t[14].cast<bool>();
					tl.objType = t[15].cast<ObjectiveFlags>();
					tl.Tmin = t[16].cast<double>();
					tl.Tmax = t[17].cast<double>();
					tl.TOFset = t[18].cast<bool>();
					tl.bnds_min = t[19].cast<py::list>();
					tl.bnds_max = t[20].cast<py::list>();
					return tl;
				}
			)
		);

		return;
	}


};