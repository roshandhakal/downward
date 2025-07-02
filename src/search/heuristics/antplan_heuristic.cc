/************************  antplan_heuristic.cc  ************************/
#include "antplan_heuristic.h"

#include "../task_proxy.h"
#include "../utils/logging.h"

#include <pybind11/embed.h>   // single-header embed interface
namespace py = pybind11;

using namespace std;
namespace antplan_heuristic {

/* ------------  Static helper to start CPython & load callback  ------------ */
void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready)
        return;

    // Start the embedded interpreter.
    py::initialize_interpreter();

    // Make sure the user's "python/" dir is on sys.path.
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, "./python");

    // Import the user script and store the cost function.
    try {
        py::module_ mdl = py::module_::import("antplan_model");
        py_cost_fn      = mdl.attr("anticipatory_cost_fn");
        py_ready        = true;
    } catch (const py::error_already_set &e) {
        cerr << "ANTPLAN: Failed to import antplan_model.py:\n" << e.what() << endl;
        throw;
    }
}

/* ----------------------  Constructor ---------------------- */
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

/* ----------------------  Main evaluation ---------------------- */
int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    // Convert ancestor_state (search copy) → global task state.
    State state = convert_ancestor_state(ancestor_state);

    /* Encode planner state as a vector<int> (one entry per variable).  */
    vector<int> values;
    values.reserve(task_proxy.get_variables().size());
    for (VariableProxy var : task_proxy.get_variables())
        values.push_back(state[var].get_value());

    /* Call Python. */
    try {
        int cost = py_cost_fn(py::cast(values)).cast<int>();
        return cost;
    } catch (const py::error_already_set &e) {
        cerr << "ANTPLAN: Python threw an exception:\n" << e.what() << endl;
        return DEAD_END;               // treat as dead end on failure
    }
}

/* -------------  Option-parser glue (registers “antplan()”) ------------- */
class AntPlanFeature
    : public plugins::TypedFeature<Evaluator, AntPlanHeuristic> {
public:
    AntPlanFeature() : TypedFeature("antplan") {
        document_title("Anticipatory-plan Python heuristic");
        add_heuristic_options_to_feature(*this, "antplan");

        document_property("admissible", "no");
        document_property("consistent", "no");
        document_property("safe",       "yes");
    }
    shared_ptr<AntPlanHeuristic>
    create_component(const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<AntPlanHeuristic>(
            get_heuristic_arguments_from_options(opts));
    }
};
static plugins::FeaturePlugin<AntPlanFeature> _plugin;

}  // namespace antplan_heuristic
