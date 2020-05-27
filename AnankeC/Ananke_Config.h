#pragma once#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Geometry>
#include <iostream>
#include "Dynamics.h"
#include "TrajLeg.h"
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>

using namespace pagmo;

struct Ananke_Config
{
	py::list TrajLegs;
	py::list LegLinks;
	double maxTOF;
	double minTOF; 
	int idxLegObj;

	void add_leg_link(int l1, int l2, VecFuncType lfun, JacFuncType dlfun, int length, py::list params, bool td) 
	{
		this->LegLinks.append(py::make_tuple(l1, l2, lfun, dlfun, length, params, td));
	}

	void set_TOF(double minTOF, double maxTOF) 
	{
		this->maxTOF = maxTOF;
		this->minTOF = minTOF;
	}

	py::list get_array_data(Eigen::VectorXd x) 
	{
		py::list out;
		double t0 = 0.0;
		for (int ii = 0; ii < py::len(this->TrajLegs); ii++)
		{
			TrajLeg TL = py::cast<TrajLeg>(this->TrajLegs[ii]);
			double T0 = 0.0;
			int idT = this->get_dvi_T(this->idxLegObj);
			double dt = x[idT] / static_cast<double>(TL.num_nodes);

			for (int jj = 0; jj < ii; jj++)
			{
				T0 += x[this->get_dvi_T(jj)];
			}
			Eigen::MatrixXd outLeg = Eigen::MatrixXd::Zero(TL.num_nodes, 1 + TL.lenN);
			for (int jj = 0; jj < TL.num_nodes; jj++)
			{
				int id0 = this->get_dvi_N(ii, jj);
				int idf = id0 + TL.lenN;
				double Tk = T0 + dt * static_cast<double>(jj);
				Eigen::VectorXd XU = x.segment(id0, TL.lenN);
				outLeg(jj, 0) = Tk;
				outLeg.block(jj, 1, 1, TL.lenN) = XU.transpose();
			}
			out.append(outLeg);
		}


		return out;
	}

	vector_double::size_type get_nec() 
	{
		int num_defects = 0;
		int num_con_user = 0;
		for (int ii = 0; ii < py::len(this->TrajLegs); ii++)
		{
			TrajLeg TL = py::cast<TrajLeg>(this->TrajLegs[ii]);
			num_defects += TL.lenX * (TL.num_nodes - 1);

			for (int jj = 0; jj < py::len(TL.coneqs_f); jj++)
			{
				py::tuple coneq = py::cast<py::tuple>(TL.coneqs_f[jj]);
				num_con_user += py::cast<int>(coneq[2]);
			}
			for (int jj = 0; jj < py::len(TL.coneqs_b); jj++)
			{
				py::tuple coneq = py::cast<py::tuple>(TL.coneqs_b[jj]);
				num_con_user += py::cast<int>(coneq[2]);
			}
			for (int jj = 0; jj < py::len(TL.coneqs_p); jj++)
			{
				py::tuple coneq = py::cast<py::tuple>(TL.coneqs_p[jj]);
				num_con_user += py::cast<int>(coneq[2]) * TL.num_nodes;
			}
		}

		for (int ii = 0; ii < py::len(this->LegLinks); ii++)
		{
			py::tuple LL = py::cast<py::tuple>(this->LegLinks[ii]);
			num_con_user += py::cast<int>(LL[4]);
		}

		return num_defects + num_con_user;
	}

	vector_double::size_type get_nic() 
	{
		int num_con_user = 0;
		for (int ii = 0; ii < py::len(this->TrajLegs); ii++)
		{
			TrajLeg TL = py::cast<TrajLeg>(this->TrajLegs[ii]);

			for (int jj = 0; jj < py::len(TL.conins_f); jj++)
			{
				py::tuple conin = py::cast<py::tuple>(TL.conins_f[jj]);
				num_con_user += py::cast<int>(conin[2]);
			}
			for (int jj = 0; jj < py::len(TL.conins_b); jj++)
			{
				py::tuple conin = py::cast<py::tuple>(TL.conins_b[jj]);
				num_con_user += py::cast<int>(conin[2]);
			}
			for (int jj = 0; jj < py::len(TL.conins_p); jj++)
			{
				py::tuple conin = py::cast<py::tuple>(TL.conins_p[jj]);
				num_con_user += py::cast<int>(conin[2]) * TL.num_nodes;
			}

			if (TL.Tmin > 0.0)
			{
				num_con_user += 2;
			}
		}

		if (this->maxTOF > 0.0)
		{
			num_con_user += 2;
		}

		return num_con_user;
	}

	std::pair<vector_double, vector_double> get_bounds()
	{
		vector_double LB;
		vector_double UB;
		for (int ii = 0; ii < py::len(this->TrajLegs); ii++)
		{

			TrajLeg TL = py::cast<TrajLeg>(this->TrajLegs[ii]);
			LB.push_back(-100000.0);
			UB.push_back(+100000.0);
			for (int jj = 0; jj < TL.num_nodes; jj++)
			{
				for (int kk = 0; kk < py::len(TL.bnds_min); kk++)
				{
					LB.push_back(py::cast<double>(TL.bnds_min[kk]));
					UB.push_back(py::cast<double>(TL.bnds_max[kk]));
				}
			}
		}
		auto rtn = std::make_pair(LB, UB);
		return rtn;
	}


	double calc_J(Eigen::VectorXd x) 
	{
		TrajLeg TLobj = py::cast<TrajLeg>(this->TrajLegs[this->idxLegObj]);
		TrajLeg TL = TLobj;
		double T0 = 0.0;
		for (int ii = 0; ii < this->idxLegObj; ii++)
		{
			T0 += x[this->get_dvi_T(ii)];
		}
		int idT = this->get_dvi_T(this->idxLegObj);
		double dt = x[idT] / static_cast<double>(TL.num_nodes);


		double J = 0.0;
		if (TL.objType == ObjectiveFlags::LAGRANGE)
		{
			J = 0.0;
			for (int ii = 0; ii < TL.num_nodes - 1; ii++)
			{
				
				double Tk = T0 + dt * static_cast<double>(ii);
				double Tkp1 = T0 + dt * static_cast<double>(ii + 1);
				int id0_Xk = this->get_dvi_N(this->idxLegObj, ii);
				int idf_Xk = id0_Xk + TL.lenX;
				int id0_Uk = idf_Xk;
				int idf_Uk = id0_Uk + TL.lenU;
				int id0_Xkp1 = idf_Uk;
				int idf_Xkp1 = id0_Xkp1 + TL.lenX;
				int id0_Ukp1 = idf_Xkp1;
				int idf_Ukp1 = id0_Ukp1 + TL.lenU;
				Eigen::VectorXd Xk = x.segment(id0_Xk, idf_Xk - id0_Xk);
				Eigen::VectorXd Uk = x.segment(id0_Uk, idf_Uk - id0_Uk);
				Eigen::VectorXd Xkp1 = x.segment(id0_Xkp1, idf_Xkp1 - id0_Xkp1);
				Eigen::VectorXd Ukp1 = x.segment(id0_Ukp1, idf_Ukp1 - id0_Ukp1);
				py::list params = py::cast<py::list>(TLobj.J[2]);
				double J1 = py::cast<ScaFuncType>(TLobj.J[0])(Xk, Uk, Tk, params);
				double J2 = py::cast<ScaFuncType>(TLobj.J[0])(Xkp1, Ukp1, Tkp1, params);
				J += 0.5 * (J1 + J2) * dt;
			}
		}
		return J;
	}

	vector_double fitness(const vector_double& dv)
	{

		vector_double dvin = dv;
		Eigen::VectorXd x(dv.size());
		for (int ii = 0; ii < dv.size(); ii++)
		{
			x[ii] = dv[ii];
		}

		Eigen::VectorXd OBJVAL = Eigen::VectorXd::Zero(1);
		Eigen::VectorXd CONEQ = Eigen::VectorXd::Zero(this->get_nec() - py::len(this->LegLinks));
		Eigen::VectorXd CONIN = Eigen::VectorXd::Zero(this->get_nic());
		Eigen::VectorXd CONLK = Eigen::VectorXd::Zero(py::len(this->LegLinks));

		// Calculate objective.
		double J = this->calc_J(x);
		OBJVAL(0) = J;

		// Constraints
		double tofTOT = 0.0;
		py::list constr_eqs;
		py::list constr_ins;
		for (int ii = 0; ii < py::len(this->TrajLegs); ii++)
		{
			TrajLeg TL = py::cast<TrajLeg>(this->TrajLegs[ii]);
			double T0 = 0.0;
			for (int ii = 0; ii < this->idxLegObj; ii++)
			{
				T0 += x[this->get_dvi_T(ii)];
			}
			int idT = this->get_dvi_T(this->idxLegObj);
			double dt = x[idT] / static_cast<double>(TL.num_nodes);
			tofTOT += x[idT];

			// Equality constraints
			for (int jj = 0; jj < py::len(TL.coneqs_f); jj++)
			{
				py::tuple coneq = py::cast<py::tuple>(TL.coneqs_f[jj]);
				int id0 = this->get_dvi_N(ii, 0);
				Eigen::VectorXd X = x.segment(id0, TL.lenX);
				Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
				double T = T0;
				py::list params = py::cast<py::list>(coneq[3]);
				constr_eqs.append(py::cast<VecFuncType>(coneq[0])(X, U, T, params));
			}
			for (int jj = 0; jj < py::len(TL.coneqs_b); jj++)
			{
				py::tuple coneq = py::cast<py::tuple>(TL.coneqs_b[jj]);
				int id0 = this->get_dvi_N(ii, -1);
				Eigen::VectorXd X = x.segment(id0, TL.lenX);
				Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
				double T = T0 + x[this->get_dvi_T(ii)];
				py::list params = py::cast<py::list>(coneq[3]);
				constr_eqs.append(py::cast<VecFuncType>(coneq[0])(X, U, T, params));
			}
			for (int jj = 0; jj < py::len(TL.coneqs_p); jj++)
			{
				py::tuple coneq = py::cast<py::tuple>(TL.coneqs_p[jj]);
				py::list params = py::cast<py::list>(coneq[3]);
				for (int kk = 0; kk < TL.num_nodes; kk++)
				{
					int id0 = this->get_dvi_N(ii, kk);
					Eigen::VectorXd X = x.segment(id0, TL.lenX);
					Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
					double T = T0 + dt * static_cast<double>(kk);
					constr_eqs.append(py::cast<VecFuncType>(coneq[0])(X, U, T, params));
				}
			}

			// Collocation constraints
			for (int jj = 0; jj < TL.num_nodes-1; jj++)
			{
				double Tk = T0 + dt * static_cast<double>(ii);
				double Tkp1 = T0 + dt * static_cast<double>(ii + 1);
				int id0_Xk = this->get_dvi_N(this->idxLegObj, ii);
				int idf_Xk = id0_Xk + TL.lenX;
				int id0_Uk = idf_Xk;
				int idf_Uk = id0_Uk + TL.lenU;
				int id0_Xkp1 = idf_Uk;
				int idf_Xkp1 = id0_Xkp1 + TL.lenX;
				int id0_Ukp1 = idf_Xkp1;
				int idf_Ukp1 = id0_Ukp1 + TL.lenU;
				py::list params = py::cast<py::list>(TL.f[3]);
				Eigen::VectorXd Xk = x.segment(id0_Xk, idf_Xk - id0_Xk);
				Eigen::VectorXd Uk = x.segment(id0_Uk, idf_Uk - id0_Uk);
				Eigen::VectorXd Xkp1 = x.segment(id0_Xkp1, idf_Xkp1 - id0_Xkp1);
				Eigen::VectorXd Ukp1 = x.segment(id0_Ukp1, idf_Ukp1 - id0_Ukp1);
				Eigen::VectorXd fk = py::cast<VecFuncType>(TL.f[0])(Xk, Uk, Tk, params);
				Eigen::VectorXd fkp1 = py::cast<VecFuncType>(TL.f[0])(Xkp1, Ukp1, Tkp1, params);
				Eigen::VectorXd Uc = 0.5 * (Uk + Ukp1);
				Eigen::VectorXd Xc = 0.5 * (Xk + Xkp1) + dt / 8 * (fk - fkp1);
				double Tc = 0.5 * (Tk + Tkp1);
				Eigen::VectorXd fc = py::cast<VecFuncType>(TL.f[0])(Xc, Uc, Tc, params);
				Eigen::VectorXd constr_p = Xk - Xkp1 + dt / 6 * (fk + 4 * fc + fkp1);
				constr_eqs.append(constr_p);
			}


			// Inequality constraints
			for (int jj = 0; jj < py::len(TL.conins_f); jj++)
			{
				py::tuple conin = py::cast<py::tuple>(TL.conins_f[jj]);
				int id0 = this->get_dvi_N(ii, 0);
				Eigen::VectorXd X = x.segment(id0, TL.lenX);
				Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
				double T = T0;
				py::list params = py::cast<py::list>(conin[3]);
				constr_ins.append(py::cast<VecFuncType>(conin[0])(X, U, T, params));
			}
			for (int jj = 0; jj < py::len(TL.conins_b); jj++)
			{
				py::tuple conin = py::cast<py::tuple>(TL.conins_b[jj]);
				int id0 = this->get_dvi_N(ii, -1);
				Eigen::VectorXd X = x.segment(id0, TL.lenX);
				Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
				double T = T0 + x[this->get_dvi_T(ii)];
				py::list params = py::cast<py::list>(conin[3]);
				constr_ins.append(py::cast<VecFuncType>(conin[0])(X, U, T, params));
			}
			for (int jj = 0; jj < py::len(TL.conins_p); jj++)
			{
				py::tuple conin = py::cast<py::tuple>(TL.conins_p[jj]);
				py::list params = py::cast<py::list>(conin[3]);
				for (int kk = 0; kk < TL.num_nodes; kk++)
				{
					int id0 = this->get_dvi_N(ii, kk);
					Eigen::VectorXd X = x.segment(id0, TL.lenX);
					Eigen::VectorXd U = x.segment(id0 + TL.lenX, TL.lenU);
					double T = T0 + dt * static_cast<double>(kk);
					constr_ins.append(py::cast<VecFuncType>(conin[0])(X, U, T, params));
				}
			}

			// TOF constraints for each leg
			Eigen::VectorXd tcon(1);
			tcon(0) = TL.Tmin - x[idT];
			constr_ins.append(tcon);
			tcon(0) = x[idT] - TL.Tmax;
			constr_ins.append(tcon);
		}
		
		// Linking constraints
		py::list constr_lks;
		for (int ii = 0; ii < py::len(this->LegLinks); ii++)
		{
			py::tuple LL = py::cast<py::tuple>(this->LegLinks[ii]);
			int li0 = py::cast<int>(LL[0]);
			int li1 = py::cast<int>(LL[1]);
			py::list params = py::cast<py::list>(LL[5]);
			int id0_Xk = this->get_dvi_N(this->idxLegObj, ii);
			int idf_Xk = id0_Xk + py::cast<TrajLeg>(this->TrajLegs[li0]).lenX;
			int id0_Uk = idf_Xk;
			int idf_Uk = id0_Uk + py::cast<TrajLeg>(this->TrajLegs[li0]).lenU;
			int id0_Xkp1 = idf_Uk;
			int idf_Xkp1 = id0_Xkp1 + py::cast<TrajLeg>(this->TrajLegs[li1]).lenX;
			int id0_Ukp1 = idf_Xkp1;
			int idf_Ukp1 = id0_Ukp1 + py::cast<TrajLeg>(this->TrajLegs[li1]).lenU;
			Eigen::VectorXd Xk = x.segment(id0_Xk, idf_Xk - id0_Xk);
			Eigen::VectorXd Uk = x.segment(id0_Uk, idf_Uk - id0_Uk);
			Eigen::VectorXd Xkp1 = x.segment(id0_Xkp1, idf_Xkp1 - id0_Xkp1);
			Eigen::VectorXd Ukp1 = x.segment(id0_Ukp1, idf_Ukp1 - id0_Ukp1);
			double T = this->get_dvi_T(li1);
			Eigen::VectorXd con = py::cast<LnkFuncType>(LL[2])(Xk, Uk, Xkp1, Ukp1, T, params);
			constr_lks.append(con);
		}

		// Total TOF constraints.
		Eigen::VectorXd tcon(1);
		tcon(0) = this->minTOF - tofTOT;
		constr_ins.append(tcon);
		tcon(0) = tofTOT - this->maxTOF;
		constr_ins.append(tcon);

		// Copy data into lists.
		int ic = 0;
		for (int ii = 0; ii < py::len(constr_eqs); ii++)
		{
			
			Eigen::VectorXd con = py::cast<Eigen::VectorXd>(constr_eqs[ii]);
			CONEQ.segment(ic, con.size()) = con;
			ic += con.size();
		}
		ic = 0;
		for (int ii = 0; ii < py::len(constr_lks); ii++)
		{
			
			Eigen::VectorXd con = py::cast<Eigen::VectorXd>(constr_lks[ii]);
			CONLK.segment(ic, con.size()) = con;
			ic += con.size();
		}
		ic = 0;
		for (int ii = 0; ii < py::len(constr_ins); ii++)
		{
			
			Eigen::VectorXd con = py::cast<Eigen::VectorXd>(constr_ins[ii]);
			CONIN.segment(ic, con.size()) = con;
			ic += con.size();
		}

		Eigen::VectorXd CONCB(CONEQ.size() + CONLK.size());
		CONCB << CONEQ, CONLK;
		Eigen::VectorXd out(1 + this->get_nec() + this->get_nic());
		out.segment(0, 1) = OBJVAL;
		out.segment(1, this->get_nec()) = CONCB;
		out.segment(1 + this->get_nec(), this->get_nic()) = CONIN;
		vector_double outRtn;
		for (int ii = 0; ii < out.size(); ii++)
		{
			outRtn.push_back(out[ii]);
		}
		return outRtn;
	}

	vector_double gradient(const vector_double& dv)
	{

		vector_double dvin = dv;
		Eigen::VectorXd x(dv.size());
		for (int ii = 0; ii < dv.size(); ii++)
		{
			x[ii] = dv[ii];
		}

		Eigen::MatrixXd est_jac = this->estimate_jacobian(x, 1e-6);
		Eigen::MatrixXd ejt = est_jac.transpose();
		Eigen::Map<Eigen::VectorXd> out(ejt.data(), ejt.size());

		vector_double outRtn;
		for (int ii = 0; ii < out.size(); ii++)
		{
			outRtn.push_back(out[ii]);
		}

		return outRtn;
	}

	Eigen::MatrixXd estimate_jacobian(Eigen::VectorXd x, double delta) 
	{
		Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(1 + this->get_nec() + this->get_nic(), x.size());
		for (int ii = 0; ii < x.size(); ii++)
		{
			Eigen::VectorXd x0 = x;
			Eigen::VectorXd x1 = x;
			x0[ii] = x0[ii] - delta;
			x1[ii] = x1[ii] + delta;
			vector_double xv0;
			vector_double xv1;
			for (int jj = 0; jj < x.size(); jj++)
			{
				xv0.push_back(x0[ii]);
				xv1.push_back(x1[ii]);
			}
			vector_double fv0 = this->fitness(xv0);
			vector_double fv1 = this->fitness(xv1);
			Eigen::VectorXd f0 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(fv0.data(), fv0.size());
			Eigen::VectorXd f1 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(fv1.data(), fv1.size());
			Eigen::VectorXd df = (f1 - f0) / (2.0 * delta);
			jac.block(0,ii,df.size(),1) = df;
		}
		return jac;
	}

	void add_leg(TrajLeg tl) 
	{
		TrajLegs.append(tl);
		if (tl.objSet)
		{
			this->idxLegObj = py::len(TrajLegs) - 1;
		}
	}

	int getTotLength() 
	{
		int len = 0;
		for (int ii = 0; ii < py::len(TrajLegs); ii++)
		{
			len += py::cast<TrajLeg>(TrajLegs[ii]).getTotLength();
		}
		return len;
	}

	int get_dvi_L(int idx_leg) 
	{
		int i0 = 0;
		if (idx_leg == -1)
		{
			idx_leg = py::len(this->TrajLegs) - 1;
		}
		for (int ii = 0; ii < idx_leg; ii++)
		{
			i0 = i0 + py::cast<TrajLeg>(this->TrajLegs[ii]).getTotLength();
		}
		return i0;
	}

	int get_dvi_N(int idx_leg, int idx_node) 
	{
		int i0 = this->get_dvi_L(idx_leg);
		i0 += 1; // Compensate for time
		if (idx_node == -1)
		{
			idx_node = py::cast<TrajLeg>(this->TrajLegs[idx_leg]).num_nodes - 1;
		}
		for (int ii = 0; ii < idx_node; ii++)
		{
			i0 = i0 + py::cast<TrajLeg>(this->TrajLegs[idx_leg]).lenN;
		}
		return i0;
	}

	int get_dvi_U(int idx_leg, int idx_node) 
	{
		int i0 = this->get_dvi_N(idx_leg, idx_node);
		i0 = i0 + py::cast<TrajLeg>(this->TrajLegs[idx_leg]).lenX;
		return i0;
	}

	int get_dvi_X(int idx_leg, int idx_node) 
	{
		int i0 = this->get_dvi_N(idx_leg, idx_node);
		return i0;
	}

	int get_dvi_T(int idx_leg)
	{
		int i0 = this->get_dvi_L(idx_leg);
		return i0;
	}

	static void Build(py::module& m)
	{
		auto obj = py::class_<Ananke_Config>(m, "Ananke_Config");
		obj.def(py::init<>());
		obj.def_readwrite("TrajLegs", &Ananke_Config::TrajLegs);
		obj.def_readwrite("LegLinks", &Ananke_Config::LegLinks);
		obj.def_readwrite("maxTOF", &Ananke_Config::maxTOF);
		obj.def_readwrite("minTOF", &Ananke_Config::minTOF);
		obj.def_readwrite("idxLegObj", &Ananke_Config::idxLegObj);

		obj.def("add_leg_link", &Ananke_Config::add_leg_link);
		obj.def("set_TOF", &Ananke_Config::set_TOF);
		obj.def("get_array_data", &Ananke_Config::get_array_data);
		obj.def("get_nec", &Ananke_Config::get_nec);
		obj.def("get_nic", &Ananke_Config::get_nic);
		obj.def("get_bounds", &Ananke_Config::get_bounds);
		obj.def("calc_J", &Ananke_Config::calc_J);
		obj.def("fitness", &Ananke_Config::fitness);
		obj.def("gradient", &Ananke_Config::gradient);
		obj.def("estimate_jacobian", &Ananke_Config::estimate_jacobian);

		obj.def("add_leg", &Ananke_Config::add_leg);
		obj.def("getTotLength", &Ananke_Config::getTotLength);
		obj.def("get_dvi_L", &Ananke_Config::get_dvi_L);
		obj.def("get_dvi_N", &Ananke_Config::get_dvi_N);
		obj.def("get_dvi_U", &Ananke_Config::get_dvi_U);
		obj.def("get_dvi_X", &Ananke_Config::get_dvi_X);
		obj.def("get_dvi_T", &Ananke_Config::get_dvi_T);

		obj.def(
			py::pickle(
				[](const Ananke_Config& ac) {
					return py::make_tuple(
						ac.TrajLegs,
						ac.LegLinks,
						ac.maxTOF,
						ac.minTOF,
						ac.idxLegObj
					);
				},
				[](py::tuple t) {
					Ananke_Config ac;
					ac.TrajLegs = py::cast<py::list>(t[0]);
					ac.LegLinks = py::cast<py::list>(t[1]);
					ac.maxTOF = py::cast<double>(t[2]);
					ac.minTOF = py::cast<double>(t[3]);
					ac.idxLegObj = py::cast<int>(t[4]);
					return ac;
				}
			)
		);

	}
};

static Ananke_Config ac;
static vector_double X0;

void set_dv(vector_double dv)
{
	X0 = dv;
}
void set_ac(Ananke_Config ac_in)
{
	ac = ac_in;
}

struct Ananke_Problem {

	vector_double::size_type get_nec() const
	{
		return ac.get_nec();
	}
	vector_double::size_type get_nic() const
	{
		return ac.get_nic();
	}
	vector_double gradient(const vector_double& dv) const
	{
		return ac.gradient(dv);
	}
	vector_double fitness(const vector_double& dv) const
	{
		return ac.fitness(dv);
	}
	std::pair<vector_double, vector_double> get_bounds() const
	{
		return ac.get_bounds();
	}
};

// Our simple example problem, version 0.
struct problem_v0 {
	// Implementation of the objective function.
	vector_double fitness(const vector_double& dv) const
	{
		return {
		  dv[0] * dv[3] * (dv[0] + dv[1] + dv[2]) + dv[2],                     // objfun
		  dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3] - 40., // equality con.
		  25. - dv[0] * dv[1] * dv[2] * dv[3]                                  // inequality con.
		};
	}
	// Implementation of the box bounds.
	std::pair<vector_double, vector_double> get_bounds() const
	{
		return { {1., 1., 1., 1.}, {5., 5., 5., 5.} };
	}
	vector_double::size_type get_nec() const
	{
		return 1;
	}
	vector_double::size_type get_nic() const
	{
		return 1;
	}

};

void optimize()
{
	std::cout << "Print problem...\n";

	// Construct a pagmo::problem from our example problem.
	problem p{ problem_v0{} };
	algorithm algo{ nlopt("cobyla") };
	algo.set_verbosity(1);
	algo.extract<nlopt>()->set_maxeval(200);
	population pop{ p, 200 };
	// pop.push_back(X0);
	pop = algo.evolve(pop);

	std::cout << pop.champion_f()[0] << std::endl;

}