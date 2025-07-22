#include "antplan_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"
#include <pybind11/stl.h> 
#include <cassert>
#include <map>
#include <algorithm>

namespace py = pybind11;
using namespace std;

namespace antplan_heuristic {

// Static definitions
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name = "anticipatory_cost_fn";

AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : AdditiveHeuristic(opts),
      relaxed_plan(task_proxy.get_operators().size(), false) {
    string func_name = opts.get<std::string>("function");
    initialize_python_function(func_name);

    utils::g_log << "Initializing AntPlan heuristic with Python function: "
                 << func_name << endl;
}

AntPlanHeuristic::~AntPlanHeuristic() {
    // Do not finalize interpreter globally; FD may use it for other components.
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
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, ".");
    py::module mdl = py::module::import("eval_anticipatory_plan"); // Python file
    py_cost_fn = mdl.attr(py_func_name.c_str());
    py_ready = true;
}

std::map<std::string, std::string> AntPlanHeuristic::convert_state_to_map(const State &state) {
    std::map<std::string, std::string> state_map;
    int num_vars = task_proxy.get_variables().size();
    for (int var_id = 0; var_id < num_vars; ++var_id) {
        VariableProxy var = task_proxy.get_variables()[var_id];
        FactProxy fact = state[var_id]; // Access fact directly
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

#include <pybind11/stl.h>  // ✅ Needed for std::map -> Python dict conversion

int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);
    int h_add = compute_add_and_ff(state);
    if (h_add == DEAD_END)
        return h_add;

    // FF part: compute h_FF and preferred operators
    for (PropID goal_id : goal_propositions)
        mark_preferred_operators_and_relaxed_plan(state, goal_id);

    int h_ff = 0;
    for (size_t op_no = 0; op_no < relaxed_plan.size(); ++op_no) {
        if (relaxed_plan[op_no]) {
            relaxed_plan[op_no] = false; // Reset
            h_ff += task_proxy.get_operators()[op_no].get_cost();
        }
    }

    // ✅ Compute anticipatory cost for the current state
    std::map<std::string, std::string> state_map;
    for (size_t var_id = 0; var_id < task_proxy.get_variables().size(); ++var_id) {
        VariableProxy var = task_proxy.get_variables()[var_id];
        FactProxy fact = state[var_id];
        state_map[var.get_name()] = fact.get_name();
    }

    int anticipatory_cost = 0;
    try {
        anticipatory_cost = py_cost_fn(py::cast(state_map)).cast<int>();
    } catch (const std::exception &e) {
        utils::g_log << "[AntPlan] Python function failed: " << e.what() << endl;
    }

    int total_h = h_ff + anticipatory_cost;

    // ✅ Debug Output
    utils::g_log << "\n[AntPlan] Current State Heuristic:" << endl;
    utils::g_log << "  h_FF = " << h_ff << endl;
    utils::g_log << "  anticipatory_cost = " << anticipatory_cost << endl;
    utils::g_log << "  total heuristic (h) = " << total_h << endl;
    utils::g_log << "  State facts:" << endl;

    for (size_t var_id = 0; var_id < task_proxy.get_variables().size(); ++var_id) {
        VariableProxy var = task_proxy.get_variables()[var_id];
        FactProxy fact = state[var_id];
        utils::g_log << "    " << var.get_name() << " = " << fact.get_name() << endl;
    }
    utils::g_log << "----------------------------------------" << endl;

    return total_h;
}

// Plugin integration
static std::shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "AntPlan heuristic",
        "Combines FF heuristic with anticipatory state evaluation using Python.");
    parser.add_option<std::string>(
        "function", "Python function name in antplan_model.py");
    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    else
        return std::make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

} // namespace antplan_heuristic
