#include "antplan_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"
#include "../algorithms/priority_queues.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

namespace antplan_heuristic {

// ===== Static definitions =====
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
string AntPlanHeuristic::py_func_name;
string AntPlanHeuristic::py_module_name;

// ===== ctor / dtor =====
AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : AdditiveHeuristic(opts),
      relaxed_plan(task_proxy.get_operators().size(), false) {
    
    // Store which heuristics to compute from options
    compute_ff = opts.get<bool>("compute_ff");
    compute_hmax = opts.get<bool>("compute_hmax");

    if (!compute_ff && !compute_hmax) {
        throw std::runtime_error("AntPlanHeuristic must be configured to compute at least one base heuristic (compute_ff or compute_hmax).");
    }

    // Python options
    py_func_name   = opts.get<string>("function");
    py_module_name = opts.get<string>("module");
    py_ready       = false;

    ensure_python_ready();
}

AntPlanHeuristic::~AntPlanHeuristic() {}

void AntPlanHeuristic::initialize_python_function(const string &func_name) {
    py_func_name = func_name;
    py_ready = false;
    ensure_python_ready();
}

void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready) return;
    if (!Py_IsInitialized()) py::initialize_interpreter();
    py::gil_scoped_acquire gil;
    try {
        if (py_module_name.empty()) throw runtime_error("No Python module provided for AntPlan.");
        py::module::import("sys").attr("path").attr("insert")(0, ".");
        py::object mdl = py::module::import(py_module_name.c_str());
        if (!py::hasattr(mdl, py_func_name.c_str())) {
            throw runtime_error("Python function '" + py_func_name + "' not found in module.");
        }
        py_cost_fn = mdl.attr(py_func_name.c_str());
        py_ready = true;
        utils::g_log << "[AntPlan] Python ready." << endl;
    } catch (const exception &e) {
        utils::g_log << "[AntPlan] Failed to initialize Python: " << e.what() << endl;
        if (PyErr_Occurred()) {
            py::object traceback = py::module::import("traceback").attr("format_exc")();
            utils::g_log << "[AntPlan][Traceback]\n" << py::cast<string>(traceback) << endl;
        }
        py_ready = false;
        throw;
    }
}

map<string, string> AntPlanHeuristic::convert_state_to_map(const State &state) {
    map<string, string> state_map;
    for (VariableProxy var : task_proxy.get_variables()) {
        FactProxy fact = state[var];
        state_map[var.get_name()] = fact.get_name();
    }
    return state_map;
}

void AntPlanHeuristic::mark_preferred_operators_and_relaxed_plan(const State &state, PropID goal_id) {
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

// ===== Main Computation =====
int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);
    map<string, string> state_map = convert_state_to_map(state);

    if (compute_hmax) {
        int h_max = 0;
        priority_queues::AdaptiveQueue<PropID> max_queue;

        auto enqueue_if_necessary = [&](PropID prop_id, int cost) {
            Proposition *prop = get_proposition(prop_id);
            if (prop->cost == -1 || prop->cost > cost) {
                prop->cost = cost;
                max_queue.push(cost, prop_id);
            }
        };

        for (Proposition &prop : propositions) { prop.cost = -1; }
        for (UnaryOperator &op : unary_operators) {
            op.unsatisfied_preconditions = op.num_preconditions;
            op.cost = op.base_cost;
            if (op.unsatisfied_preconditions == 0) {
                enqueue_if_necessary(op.effect, op.base_cost);
            }
        }
        for (FactProxy fact : state) { enqueue_if_necessary(get_prop_id(fact), 0); }

        while (!max_queue.empty()) {
            pair<int, PropID> top_pair = max_queue.pop();
            int distance = top_pair.first;
            PropID prop_id = top_pair.second;
            int prop_cost = get_proposition(prop_id)->cost;
            if (prop_cost < distance) continue;

            for (OpID op_id : precondition_of_pool.get_slice(
                 get_proposition(prop_id)->precondition_of,
                 get_proposition(prop_id)->num_precondition_occurences)) {
                UnaryOperator *op = get_operator(op_id);
                op->cost = max(op->cost, op->base_cost + prop_cost);
                --op->unsatisfied_preconditions;
                if (op->unsatisfied_preconditions == 0) {
                    enqueue_if_necessary(op->effect, op->cost);
                }
            }
        }
        for (PropID goal_id : goal_propositions) {
            int goal_cost = get_proposition(goal_id)->cost;
            if (goal_cost == -1) return DEAD_END;
            h_max = max(h_max, goal_cost);
        }
        state_map["__h_max__"] = to_string(h_max);
    }

    if (compute_ff) {
        int h_add = compute_add_and_ff(state);
        if (h_add == DEAD_END) return DEAD_END;
        
        state_map["__h_add__"] = to_string(h_add); // Pass h_add since it's free
        
        for (PropID goal_id : goal_propositions) {
            mark_preferred_operators_and_relaxed_plan(state, goal_id);
        }
        int h_ff = 0;
        for (size_t op_no = 0; op_no < relaxed_plan.size(); ++op_no) {
            if (relaxed_plan[op_no]) {
                relaxed_plan[op_no] = false;
                h_ff += task_proxy.get_operators()[op_no].get_cost();
            }
        }
        state_map["__h_ff__"] = to_string(h_ff);
    }
    
    int final_heuristic_value = 0;
    if (py_ready) {
        try {
            py::gil_scoped_acquire gil;
            final_heuristic_value = py_cost_fn(py::cast(state_map)).cast<int>();
        } catch (const exception &e) {
            utils::g_log << "[AntPlan] Python function failed: " << e.what() << endl;
            if (PyErr_Occurred()) {
                py::object tb = py::module::import("traceback").attr("format_exc")();
                utils::g_log << py::cast<string>(tb) << endl;
            }
            return DEAD_END;
        }
    } else {
        utils::g_log << "[AntPlan] Python not ready; cannot compute." << endl;
        return DEAD_END;
    }
    return (final_heuristic_value < 0) ? DEAD_END : final_heuristic_value;
}

// ===== plugin registration =====
static shared_ptr<Heuristic> _parse(options::OptionParser &parser) {
    parser.document_synopsis(
        "AntPlan heuristic",
        "Computes selected base heuristics and passes them to a Python cost function.");

    parser.add_option<string>("function", "Python function name to call.", "distance_based_probabilistic");
    parser.add_option<string>("module", "Python module to import.", "");
    
    // Add the boolean flags to control heuristic computation
    parser.add_option<bool>("compute_ff", "Set to true to compute and pass h_ff.", "false");
    parser.add_option<bool>("compute_hmax", "Set to true to compute and pass h_max.", "false");

    Heuristic::add_options_to_parser(parser);
    options::Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    return make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

} // namespace antplan_heuristic