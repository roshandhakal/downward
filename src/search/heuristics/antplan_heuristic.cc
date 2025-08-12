#include "antplan_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <utility>

#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

namespace antplan_heuristic {

// ===== Static definitions =====
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name = "anticipatory_cost_fn";
std::string AntPlanHeuristic::py_module_name = "antplan.scripts.eval_antplan_gripper";
std::string AntPlanHeuristic::py_file_path;

// ===== ctor / dtor =====
AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : AdditiveHeuristic(opts),
      relaxed_plan(task_proxy.get_operators().size(), false) {

    // required: function (name of Python function to call inside the module/file)
    std::string func_name = opts.get<std::string>("function");
    initialize_python_function(func_name);

    // optional: module or filepath; either can be provided
    if (opts.contains("module"))
        py_module_name = opts.get<std::string>("module");
    if (opts.contains("filepath"))
        py_file_path = opts.get<std::string>("filepath");

    utils::g_log << "Initializing AntPlan heuristic\n"
                 << "  function : " << py_func_name << "\n"
                 << "  module   : " << (py_module_name.empty() ? "<none>" : py_module_name) << "\n"
                 << "  filepath : " << (py_file_path.empty() ? "<none>" : py_file_path) << std::endl;
}

AntPlanHeuristic::~AntPlanHeuristic() {
    // Do not finalize the interpreter here (FD or other components might share it).
}

// ===== helpers =====
void AntPlanHeuristic::add_to_sys_path_front(const std::string &path) {
    if (path.empty()) return;
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, path);
}

void AntPlanHeuristic::initialize_python_function(const std::string &func_name) {
    py_func_name = func_name;
    py_ready = false;
    ensure_python_ready();
}

void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready)
        return;

    // Initialize interpreter if needed
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    // We support two ways to supply the code:
    // 1) py_file_path: load a module from an explicit .py file
    // 2) py_module_name: import by module name via sys.path
    // If both are provided, filepath takes precedence.

    try {
        py::object mdl;

        if (!py_file_path.empty()) {
            // Load from an explicit file path using importlib
            py::module importlib = py::module::import("importlib.util");
            py::object spec_from_file_location = importlib.attr("spec_from_file_location");
            py::object module_from_spec = py::module::import("importlib").attr("util").attr("module_from_spec");

            // Pick a stable synthetic module name from file path
            // (Python allows arbitrary names for file-based specs)
            std::string synth_name = std::string("antplan_dynamic_") + std::to_string(std::hash<std::string>{}(py_file_path));

            py::object spec = spec_from_file_location(synth_name.c_str(), py_file_path.c_str());
            if (spec.is_none()) {
                throw std::runtime_error("importlib.util.spec_from_file_location returned None");
            }
            py::object module = module_from_spec(spec);
            spec.attr("loader").attr("exec_module")(module);
            mdl = module;

        } else {
            // Import by module name; allow CWD first (like original)
            add_to_sys_path_front(".");
            mdl = py::module::import(py_module_name.c_str());
        }

        // Lookup the function by name
        if (!py::hasattr(mdl, py_func_name.c_str())) {
            throw std::runtime_error("Python object '" + py_func_name + "' not found in module");
        }
        py_cost_fn = mdl.attr(py_func_name.c_str());
        py_ready = true;

    } catch (const std::exception &e) {
        utils::g_log << "[AntPlan] Failed to initialize Python: " << e.what() << std::endl;
        // Keep py_ready=false to prevent calls; heuristic will return 0 then.
    }
}

std::map<std::string, std::string>
AntPlanHeuristic::convert_state_to_map(const State &state) {
    std::map<std::string, std::string> state_map;
    int num_vars = task_proxy.get_variables().size();
    for (int var_id = 0; var_id < num_vars; ++var_id) {
        VariableProxy var = task_proxy.get_variables()[var_id];
        FactProxy fact = state[var_id];
        state_map[var.get_name()] = fact.get_name();
    }
    return state_map;
}

void AntPlanHeuristic::mark_preferred_operators_and_relaxed_plan(
    const State &state, PropID goal_id) {
    Proposition *goal = get_proposition(goal_id);
    if (!goal->marked) {
        goal->marked = true;
        OpID op_id = goal->reached_by;
        if (op_id != NO_OP) {
            UnaryOperator *unary_op = get_operator(op_id);
            bool is_preferred = true;
            for (PropID precond : get_preconditions(op_id)) {
                mark_preferred_operators_and_relaxed_plan(state, precond);
                if (get_proposition(precond)->reached_by != NO_OP) {
                    is_preferred = false;
                }
            }
            int operator_no = unary_op->operator_no;
            if (operator_no != -1) {
                relaxed_plan[operator_no] = true;
                if (is_preferred) {
                    OperatorProxy op = task_proxy.get_operators()[operator_no];
                    assert(task_properties::is_applicable(op, state));
                    set_preferred(op);
                }
            }
        }
    }
}

// ===== main computation =====
int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);

    // Build a simple { var_name -> fact_name } snapshot
    std::map<std::string, std::string> state_map = convert_state_to_map(state);

    int anticipatory_cost = 0;
    if (py_ready) {
        try {
            anticipatory_cost = py_cost_fn(py::cast(state_map)).cast<int>();
        } catch (const std::exception &e) {
            utils::g_log << "[AntPlan] Python function failed: " << e.what() << std::endl;
        }
    } else {
        utils::g_log << "[AntPlan] Python not ready; returning 0." << std::endl;
    }

    utils::g_log << "\n[AntPlan] Current State Heuristic:\n"
                 << "  anticipatory_cost = " << anticipatory_cost << "\n";

    return anticipatory_cost;
}

// ===== plugin registration =====
static std::shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "AntPlan heuristic",
        "Evaluates a state with a Python cost function (anticipatory cost). "
        "You can provide either a Python module name or a file path.");

    // Name of the function to call; e.g., 'anticipatory_cost_fn'
    parser.add_option<std::string>(
        "function",
        "Python function name to call (attribute in module/file).");

    // Either provide the module name...
    parser.add_option<std::string>(
        "module",
        "Python module name to import (e.g., 'pkg.subpkg.module'). "
        "Default: 'antplan.scripts.eval_antplan_gripper'.",
        "antplan.scripts.eval_antplan_gripper");

    // ...or provide a file path.
    parser.add_option<std::string>(
        "filepath",
        "Absolute or relative path to a .py file to load (takes precedence over 'module').",
        "");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    return std::make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

} // namespace antplan_heuristic
