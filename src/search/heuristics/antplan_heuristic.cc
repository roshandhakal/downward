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
// default keeps backward-compat unless you override from CLI
std::string AntPlanHeuristic::py_module_name = "antplan.scripts.eval_antplan_gripper";
std::string AntPlanHeuristic::py_file_path;

// ===== ctor / dtor =====
AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : AdditiveHeuristic(opts),
      relaxed_plan(task_proxy.get_operators().size(), false) {

    // All three are optional (defaults set in parser)
    std::string func_name = opts.get<std::string>("function");
    std::string mod_name  = opts.get<std::string>("module");
    std::string file_path = opts.get<std::string>("filepath");

    py_func_name    = func_name;
    py_module_name  = mod_name;
    py_file_path    = file_path;
    py_ready        = false;

    utils::g_log << "Initializing AntPlan heuristic\n"
                 << "  function : " << py_func_name << "\n"
                 << "  module   : " << (py_module_name.empty() ? "<none>" : py_module_name) << "\n"
                 << "  filepath : " << (py_file_path.empty() ? "<none>" : py_file_path) << std::endl;

    ensure_python_ready();
}

AntPlanHeuristic::~AntPlanHeuristic() {
    // Do not finalize interpreter here (other components might use it).
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

    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    try {
        py::object mdl;

        if (!py_file_path.empty()) {
            // Load a module from an explicit .py file
            py::module importlib_util = py::module::import("importlib.util");
            py::module importlib      = py::module::import("importlib");

            py::object spec_from_file_location = importlib_util.attr("spec_from_file_location");
            py::object module_from_spec        = importlib.attr("util").attr("module_from_spec");

            // Stable synthetic module name derived from file path
            std::string synth_name = std::string("antplan_dynamic_") +
                                     std::to_string(std::hash<std::string>{}(py_file_path));

            py::object spec = spec_from_file_location(synth_name.c_str(), py_file_path.c_str());
            if (spec.is_none()) {
                throw std::runtime_error("importlib.util.spec_from_file_location returned None");
            }
            py::object module = module_from_spec(spec);
            spec.attr("loader").attr("exec_module")(module);
            mdl = module;

        } else {
            // Import by module name; keep CWD first like original behavior
            add_to_sys_path_front(".");
            if (py_module_name.empty()) {
                throw std::runtime_error("No module or filepath provided for AntPlan.");
            }
            mdl = py::module::import(py_module_name.c_str());
        }

        if (!py::hasattr(mdl, py_func_name.c_str())) {
            throw std::runtime_error("Python object '" + py_func_name + "' not found in module");
        }
        py_cost_fn = mdl.attr(py_func_name.c_str());
        py_ready = true;

    } catch (const std::exception &e) {
        utils::g_log << "[AntPlan] Failed to initialize Python: " << e.what() << std::endl;
        py_ready = false;
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

    // Build { var_name -> fact_name }
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
        "Provide either a Python module name or a file path. Both are optional; "
        "if both are given, 'filepath' takes precedence.");

    // All of these have defaults so FD won't require them.
    parser.add_option<std::string>(
        "function",
        "Python function name to call (attribute in module/file).",
        "anticipatory_cost_fn");

    parser.add_option<std::string>(
        "module",
        "Python module name to import (e.g., 'pkg.subpkg.module').",
        "antplan.scripts.eval_antplan_gripper");

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
