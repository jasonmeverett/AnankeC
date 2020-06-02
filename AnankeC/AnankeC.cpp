#include "ex1_setup.h"
#include "TrajLeg.h"
#include "Ananke_Config.h"
#include <pagmo_plugins_nonfree/pagmo_plugins_nonfree.hpp>
#include <iostream>
#include <string>
#include <algorithm>

static Ananke_Config ac;
static vector_double X0;
using namespace std;

void set_dv(vector_double dv)
{
	X0 = dv;
}
void set_ac(Ananke_Config ac_in)
{
	ac = ac_in;
	ac.set_nec();
	ac.set_nic();
	ac.set_bounds();
	ac.calc_gradient_sparsity(X0);
}

struct Ananke_Problem {

	vector_double::size_type get_nec() const
	{
		return ac.cur_nec;
	}
	vector_double::size_type get_nic() const
	{
		return ac.cur_nic;
	}
	vector_double gradient(const vector_double& dv) const
	{
		return ac.gradient(dv);
	}
	sparsity_pattern gradient_sparsity() const
	{
		sparsity_pattern sp = ac.gradient_sparsity();
		return sp;
	}
	vector_double fitness(const vector_double& dv) const
	{
		return ac.fitness(dv);
	}
	std::pair<vector_double, vector_double> get_bounds() const
	{
		return ac.cur_bounds;
	}
};

py::tuple optimize(int max_eval, int verb, double ctol, double elw)
{
	try
	{
		// Construct a pagmo::problem from our example problem.
		problem p{ Ananke_Problem{} };
		p.set_c_tol(ctol);
		char* snopt_dll = getenv("SNOPT_DLL");
		string snopt_dll_str = snopt_dll;
		replace(snopt_dll_str.begin(), snopt_dll_str.end(), '\\', '/');
		cout << snopt_dll_str << "\n";
		if (snopt_dll == nullptr)
		{
			std::cerr << "ERROR: SNOPT_DLL Environment Variable not set.\n";
			return py::make_tuple(0.0, 0.0);
		}
		auto uda = ppnf::snopt7(true, snopt_dll, 7U);
		std::cout << uda.get_name() << std::endl;
		uda.set_integer_option("Major iterations limit", max_eval);
		uda.set_integer_option("Minor iterations limit", max_eval);
		uda.set_integer_option("Iterations limit", max_eval);
		uda.set_numeric_option("Elastic weight", elw);
		algorithm algo{ uda };
		// algo.set_verbosity(1);
		population pop{ p };
		pop.push_back(X0);
		pop = algo.evolve(pop);
		std::cout << pop.champion_f()[0] << std::endl;

		vector_double F = pop.champion_f();
		vector_double X = pop.champion_x();

		std::cout << pop << std::endl;

		return py::make_tuple(X, F);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
		return py::make_tuple(0.0, 0.0);
	}
	
}


PYBIND11_MODULE(AnankeC, m) {
    m.doc() = "AnankeC plugin"; // optional module docstring
           
    Ex1::Build(m);
    TrajLeg::Build(m);

    Ananke_Config::Build(m);
    m.def("optimize", optimize);
    m.def("set_ac", set_ac);
    m.def("set_dv", set_dv);
}