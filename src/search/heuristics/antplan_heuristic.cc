#include "antplan_heuristic.h"
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/iostream.h>
#include <unistd.h>

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"
#include "../utils/logging.h"

using namespace std;

namespace py = pybind11;

namespace antplan_heuristic {

// Static definitions
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name = "anticipatory_cost_fn";

AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : Heuristic(opts) {
    string func_name = opts.get<std::string>("function");
    initialize_python_function(func_name);

    utils::g_log << "Initialized AntPlan heuristic with Python function: " << func_name << endl;
}

AntPlanHeuristic::~AntPlanHeuristic() {
    // Leave interpreter running; don't finalize unless you're the last component using Python
}

void AntPlanHeuristic::initialize_python_function(const std::string &func_name) {
    py_func_name = func_name;
    py_ready = false;
    ensure_python_ready();
}

void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready)
        return;

    py::initialize_interpreter();
    // py::gil_scoped_acquire acquire;

    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, ".");
    py::list sys_path = sys.attr("path");

    // std::cerr << "Python sys.path:" << std::endl;
    // for (auto item : sys_path) {
    //     std::cerr << "  " << std::string(py::str(item)) << std::endl;
    // }

    try {
        py::module mdl = py::module::import("antplan_model");
        py_cost_fn = mdl.attr(py_func_name.c_str());
        py_ready = true;
    } catch (const py::error_already_set &e) {
        cerr << "ANTPLAN: Failed to load Python model or function:\n" << e.what() << endl;
        throw;
    }
}


int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    // cerr << "ANTPLAN: C++ heuristic running inside PID: " << getpid() << endl;
    State state = convert_ancestor_state(ancestor_state);

    std::map<std::string, std::string> state_map;

    for (VariableProxy var : task_proxy.get_variables()) {
        FactProxy fact = state[var];
        std::string var_name = var.get_name();
        std::string val_name = fact.get_name();
        state_map[var_name] = val_name;
    }
    // //Print state information
    // cerr << "State ID: " << state.get_id() << endl;
    // const vector<int> &variable_values = state.get_unpacked_values();
    // cerr << "Number of variables: " << variable_values.size() << endl;
    // cerr << "Variable values: ";
    // for (int value : variable_values) {
    //     cout << value << " ";
    // }
    // cerr << endl;

    // // Convert variable values to predicates
    // cerr << "Predicates: " << endl;
    // TaskProxy task = state.get_task();
    // for (VariableProxy var : task.get_variables()) {
    //     int value = state[var].get_value();
    //     string predicate = var.get_fact(value).get_name();
    //     cerr << predicate << endl;
    // }

    // Print symbolic state
    // cerr << "ANTPLAN: Symbolic State = {" << endl;
    // for (const auto &pair : state_map) {
    //     cerr << "  " << pair.first << ": " << pair.second << endl;
    // }
    // cerr << "}" << endl;

    // Pass symbolic state to Python as a dictionary
    try {
        int cost = py_cost_fn(py::cast(state_map)).cast<int>();
        cerr << "ANTPLAN: → Python returned cost = " << cost << endl;
        return cost;
    } catch (const py::error_already_set &e) {
        cerr << "ANTPLAN: Python cost function threw an error:\n" << e.what() << endl;
        return DEAD_END;
    }
}


// Plugin registration
static shared_ptr<Heuristic> _parse(options::OptionParser &parser) {
    parser.document_synopsis("Anticipatory heuristic (Python-based)", "");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "yes");

    Heuristic::add_options_to_parser(parser);
    parser.add_option<std::string>("function", "Name of the Python function to call", "anticipatory_cost_fn");

    options::Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    return make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

}
