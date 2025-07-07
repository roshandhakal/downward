/************************  antplan_heuristic.cc  ************************/
#include "antplan_heuristic.h"

#include "../task_proxy.h"
#include "../utils/logging.h"

#include <pybind11/embed.h>
namespace py = pybind11;

using namespace std;

namespace antplan_heuristic {

// Static variable definitions
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name = "anticipatory_cost_fn";

/* ------------  Python interpreter setup  ------------ */
void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready)
        return;

    py::initialize_interpreter();

    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, "./models");

    try {
        py::module mdl = py::module::import("antplan_model");
        py_cost_fn     = mdl.attr(py_func_name.c_str());
        py_ready       = true;
    } catch (const py::error_already_set &e) {
        cerr << "ANTPLAN: Failed to import antplan_model.py or function '" << py_func_name << "':\n"
             << e.what() << endl;
        throw;
    }
}

/* ------------------ Constructor ------------------ */
AntPlanHeuristic::AntPlanHeuristic(
        const std::shared_ptr<AbstractTask> &task,
        bool cache_estimates,
        const string &description,
        utils::Verbosity verbosity)
    : Heuristic(task,
                cache_estimates,
                description.empty() ? "antplan" : description,
                verbosity) {
    ensure_python_ready();
    if (log.is_at_least_normal())
        log << "Initialized AntPlan heuristic." << endl;
}

/* ------------------ Set function name ------------------ */
void AntPlanHeuristic::initialize_python_function(const std::string &func_name) {
    py_func_name = func_name;
    py_ready = false; // reset interpreter to pick new function
}

/* ------------------ Main evaluation ------------------ */
int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);

    vector<int> values;
    values.reserve(task_proxy.get_variables().size());
    for (VariableProxy var : task_proxy.get_variables())
        values.push_back(state[var].get_value());

    try {
        int cost = py_cost_fn(py::cast(values)).cast<int>();
        return cost;
    } catch (const py::error_already_set &e) {
        cerr << "ANTPLAN: Python exception in cost function:\n" << e.what() << endl;
        return DEAD_END;
    }
}

/* ------------- Plugin registration for "antplan()" ------------- */
class AntPlanFeature
    : public plugins::TypedFeature<Evaluator, AntPlanHeuristic> {
public:
    AntPlanFeature() : TypedFeature("antplan") {
        document_title("Anticipatory-plan Python heuristic");
        add_heuristic_options_to_feature(*this, "antplan");

        add_option<std::string>("function", "Name of the Python cost function to call", "anticipatory_cost_fn");


        document_property("admissible", "no");
        document_property("consistent", "no");
        document_property("safe",       "yes");
    }

    shared_ptr<AntPlanHeuristic>
    create_component(const plugins::Options &opts) const override {
        auto heuristic = plugins::make_shared_from_arg_tuples<AntPlanHeuristic>(
            get_heuristic_arguments_from_options(opts));
        std::string func_name = opts.get<std::string>("function");
        heuristic->initialize_python_function(func_name);
        return heuristic;
    }
};

static plugins::FeaturePlugin<AntPlanFeature> _plugin;

}  // namespace antplan_heuristic
