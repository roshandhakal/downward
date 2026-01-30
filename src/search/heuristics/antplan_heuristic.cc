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
std::string AntPlanHeuristic::py_func_name  = "anticipatory_cost_fn";
std::string AntPlanHeuristic::py_module_name = "antplan.scripts.eval_antplan_gripper";

// ===== ctor / dtor =====
AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : AdditiveHeuristic(opts),
      relaxed_plan(task_proxy.get_operators().size(), false) {

    // Only two options: function + module
    std::string func_name = opts.get<std::string>("function");
    std::string mod_name  = opts.get<std::string>("module");

    py_func_name   = func_name;
    py_module_name = mod_name;
    py_ready       = false;

    utils::g_log << "[AntPlan] ctor: function=" << py_func_name
                 << " module=" << (py_module_name.empty() ? "<none>" : py_module_name)
                 << std::endl;

    ensure_python_ready();
}

AntPlanHeuristic::~AntPlanHeuristic() {
    // Do not finalize interpreter here (other components might use it).
}

// ===== helpers =====
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

    py::gil_scoped_acquire gil;

    try {
        // utils::g_log << "[AntPlan] ensure_python_ready: func=" << py_func_name
        //              << " module=" << (py_module_name.empty() ? "<none>" : py_module_name)
        //              << std::endl;

        if (py_module_name.empty())
            throw std::runtime_error("No Python module provided for AntPlan.");

        py::module sys = py::module::import("sys");

        // Always keep CWD on path[0] to help when running from project root
        sys.attr("path").attr("insert")(0, ".");

        // // --- Diagnostics before import
        // utils::g_log << "[AntPlan][PyDiag] sys.version=" 
        //              << py::cast<std::string>(sys.attr("version")) << "\n";
        // utils::g_log << "[AntPlan][PyDiag] sys.executable=" 
        //              << py::cast<std::string>(sys.attr("executable")) << "\n";
        // utils::g_log << "[AntPlan][PyDiag] sys.prefix=" 
        //              << py::cast<std::string>(sys.attr("prefix")) << "\n";
        // utils::g_log << "[AntPlan][PyDiag] sys.base_prefix=" 
        //              << py::cast<std::string>(sys.attr("base_prefix")) << "\n";

        // py::list p = sys.attr("path");
        // utils::g_log << "[AntPlan][PyDiag] sys.path:\n";
        // for (size_t i = 0; i < py::len(p); ++i) {
        //     utils::g_log << "  [" << i << "] " << py::cast<std::string>(p[i]) << "\n";
        // }
        // utils::g_log << std::flush;

        // --- Import module by name only
        py::object mdl = py::module::import(py_module_name.c_str());

        if (!py::hasattr(mdl, py_func_name.c_str())) {
            py::list names = mdl.attr("__dict__").attr("keys")();
            std::string have;
            for (auto &n : names) have += py::cast<std::string>(n) + " ";
            throw std::runtime_error(
                "Python object '" + py_func_name + "' not found in module. Have: " + have);
        }

        py_cost_fn = mdl.attr(py_func_name.c_str());
        py_ready = true;
        utils::g_log << "[AntPlan] Python ready.\n";

    } catch (const std::exception &e) {
        try {
            py::object traceback = py::module::import("traceback").attr("format_exc")();
            utils::g_log << "[AntPlan] Failed to initialize Python: " << e.what() << "\n"
                         << "[AntPlan][Traceback]\n" << py::cast<std::string>(traceback) << std::endl;

            // Re-dump sys.path in the catch too, in case it changed
            try {
                py::module sys2 = py::module::import("sys");
                py::list p2 = sys2.attr("path");
                utils::g_log << "[AntPlan][PyDiag-after-fail] sys.path:\n";
                for (size_t i = 0; i < py::len(p2); ++i) {
                    utils::g_log << "  [" << i << "] " << py::cast<std::string>(p2[i]) << "\n";
                }
            } catch (...) {}
        } catch (...) {
            utils::g_log << "[AntPlan] Failed to initialize Python: " << e.what() << std::endl;
        }
        py_ready = false;
        // Optional: uncomment to fail fast instead of silently returning 0 from compute_heuristic
        // throw;
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
    std::map<std::string, std::string> state_map = convert_state_to_map(state);

    int anticipatory_cost = 0;

    if (py_ready) {
        try {
            py::gil_scoped_acquire gil;  // hold GIL during Python call
            anticipatory_cost = py_cost_fn(py::cast(state_map)).cast<int>();
        } catch (const std::exception &e) {
            // Try to dump traceback too
            try {
                py::gil_scoped_acquire gil2;
                py::object tb = py::module::import("traceback").attr("format_exc")();
                utils::g_log << "[AntPlan] Python function failed: " << e.what() << "\n"
                             << py::cast<std::string>(tb) << std::endl;
            } catch (...) {
                utils::g_log << "[AntPlan] Python function failed: " << e.what() << std::endl;
            }
        }
    } else {
        utils::g_log << "[AntPlan] Python not ready; returning 0." << std::endl;
    }

    utils::g_log << "  State facts:" << endl;

    for (size_t var_id = 0; var_id < task_proxy.get_variables().size(); ++var_id) {
        VariableProxy var = task_proxy.get_variables()[var_id];
        FactProxy fact = state[var_id];
        utils::g_log << "    " << var.get_name() << " = " << fact.get_name() << endl;
    }
    
    utils::g_log << "----------------------------------------" << endl;
    utils::g_log << "\n[AntPlan] Current State Heuristic:\n"
                 << "  anticipatory_cost = " << anticipatory_cost << "\n";

    return anticipatory_cost;
}

// ===== plugin registration =====
static std::shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "AntPlan heuristic",
        "Evaluates a state with a Python cost function (anticipatory cost). "
        "Imports strictly by Python module name and looks up a function.");

    // Only these two options remain.
    parser.add_option<std::string>(
        "function",
        "Python function name to call (attribute in module).",
        "anticipatory_cost_fn");

    parser.add_option<std::string>(
        "module",
        "Python module name to import (e.g., 'pkg.subpkg.module').",
        "antplan.scripts.eval_antplan_gripper");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    return std::make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

} // namespace antplan_heuristic