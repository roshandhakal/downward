#include "antplan_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"
#include "../utils/logging.h"

#include <map>
#include <string>
#include <vector>
#include <cassert>

using namespace std;
namespace py = pybind11;

namespace antplan_heuristic {

// Static variables
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name = "anticipatory_cost_fn";

// Constructor
AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : FFHeuristic(opts) {
    string func_name = opts.get<std::string>("function");
    initialize_python_function(func_name);
    utils::g_log << "Initialized AntPlan heuristic with Python function: "
                 << func_name << endl;
}

AntPlanHeuristic::~AntPlanHeuristic() {
    // Do not finalize Python interpreter; shared resource
}

// Initialize Python function
void AntPlanHeuristic::initialize_python_function(const std::string &func_name) {
    py_func_name = func_name;
    py_ready = false;
    ensure_python_ready();
}

// Make ensure_python_ready static
void AntPlanHeuristic::ensure_python_ready() {
    if (!py_ready) {
        py::initialize_interpreter();
        py::module main = py::module::import("__main__");
        py::object global = main.attr("__dict__");

        if (!global.contains(py_func_name.c_str())) {
            throw std::runtime_error("Python function " + py_func_name + " not found.");
        }

        py_cost_fn = global[py_func_name.c_str()];
        py_ready = true;
    }
}

// Compute AntPlan cost via Python
int AntPlanHeuristic::compute_antplan_cost(const State &state) {
    std::map<std::string, std::string> state_map;

    // Convert FD state to map<string,string>
    for (FactProxy fact : state) {
        std::string var_name = fact.get_variable().get_name();
        std::string value_name = fact.get_variable().get_fact(fact.get_value()).get_name();
        state_map[var_name] = value_name;
    }

    int ant_cost = 0;
    try {
        py::gil_scoped_acquire acquire;
        py::dict py_state;
        for (auto &pair : state_map) {
            py_state[py::str(pair.first)] = py::str(pair.second);
        }
        ant_cost = py_cost_fn(py_state).cast<int>();
    } catch (const std::exception &e) {
        utils::g_log << "[AntPlanHeuristic] Python error: " << e.what() << endl;
    }
    return ant_cost;
}

// Compute combined heuristic = FF + AntPlan
int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);

    int h_ff = FFHeuristic::compute_heuristic(state);
    if (h_ff == DEAD_END)
        return DEAD_END;

    int h_ant = compute_antplan_cost(state);
    return h_ff + h_ant;
}

// Register plugin
static shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis("AntPlan + FF heuristic combined", "");
    parser.add_option<std::string>("function", "Name of Python cost function");
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    else
        return make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

}
