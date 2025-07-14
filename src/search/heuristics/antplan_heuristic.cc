// antplan_heuristic.cc (Anticipatory exploration heuristic inspired by hmax)

#include "antplan_heuristic.h"

#include "relaxation_heuristic.h"
#include "../task_proxy.h"
#include "../utils/logging.h"
#include "../option_parser.h"
#include "../plugin.h"

#include <pybind11/stl.h>
#include <pybind11/embed.h>

#include <limits>

using namespace std;
namespace py = pybind11;

namespace antplan_heuristic {

py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name = "anticipatory_cost_fn";

AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : relaxation_heuristic::RelaxationHeuristic(opts), queue() {
    std::string func_name = opts.get<std::string>("function");
    initialize_python_function(func_name);
    utils::g_log << "Initialized AntPlan heuristic with Python cost function." << endl;
}

void AntPlanHeuristic::setup_exploration_queue() {
    queue.clear();
    for (Proposition &prop : propositions)
        prop.cost = -1;

    for (UnaryOperator &op : unary_operators) {
        op.unsatisfied_preconditions = op.num_preconditions;
        op.cost = op.base_cost;
        if (op.unsatisfied_preconditions == 0)
            enqueue_if_necessary(op.effect, op.base_cost);
    }
}

void AntPlanHeuristic::setup_exploration_queue_state(const State &state) {
    for (FactProxy fact : state) {
        PropID prop_id = get_prop_id(fact);
        enqueue_if_necessary(prop_id, 0);
    }
}

void AntPlanHeuristic::enqueue_if_necessary(PropID prop_id, int cost) {
    assert(cost >= 0);
    Proposition *prop = get_proposition(prop_id);
    if (prop->cost == -1 || prop->cost > cost) {
        prop->cost = cost;
        queue.push(cost, prop_id);
    }
    assert(prop->cost != -1 && prop->cost <= cost);
}

int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);
    setup_exploration_queue();
    setup_exploration_queue_state(state);

    int unsolved_goals = goal_propositions.size();

    while (!queue.empty()) {
        pair<int, PropID> top_pair = queue.pop();
        int distance = top_pair.first;
        PropID prop_id = top_pair.second;
        Proposition *prop = get_proposition(prop_id);
        int prop_cost = prop->cost;
        if (prop_cost < distance) continue;

        if (prop->is_goal && --unsolved_goals == 0) {
            break;
        }

        for (OpID op_id : precondition_of_pool.get_slice(
                 prop->precondition_of, prop->num_precondition_occurences)) {
            UnaryOperator *unary_op = get_operator(op_id);
            unary_op->cost = max(unary_op->cost, unary_op->base_cost + prop_cost);
            --unary_op->unsatisfied_preconditions;
            if (unary_op->unsatisfied_preconditions == 0)
                enqueue_if_necessary(unary_op->effect, unary_op->cost);
        }
    }

    // If any goal is unreachable, it's a dead end
    for (PropID goal_id : goal_propositions) {
        const Proposition *goal = get_proposition(goal_id);
        if (goal->cost == -1)
            return DEAD_END;
    }

    // Compute symbolic cost like hmax
    int symbolic_cost = 0;
    for (PropID goal_id : goal_propositions) {
        const Proposition *goal = get_proposition(goal_id);
        symbolic_cost = max(symbolic_cost, goal->cost);
    }

    // Build relaxed state from all propositions with cost != -1
    std::map<std::string, std::string> relaxed_state_map;
    int idx = 0;
    for (const Proposition &prop : propositions) {
        if (prop.cost != -1) {
            // Convert this proposition to its fact name using get_fact(idx)
            FactPair fact_pair = get_fact(idx);
            std::string fact_name = task_proxy.get_variables()[fact_pair.var]
                                                 .get_fact(fact_pair.value)
                                                 .get_name();
            relaxed_state_map["p" + std::to_string(idx)] = fact_name;
        }
        idx++;
    }

    int anticipatory_cost = 0;
    try {
        anticipatory_cost = py_cost_fn(py::cast(relaxed_state_map)).cast<int>();
    } catch (const py::error_already_set &e) {
        cerr << "ANTPLAN: Python error in symbolic evaluation:\n" << e.what() << endl;
        return DEAD_END;
    }

    int final_cost = symbolic_cost + anticipatory_cost;

    cerr << "[ANTPLAN DEBUG] symbolic=" << symbolic_cost
         << ", anticipatory=" << anticipatory_cost
         << ", total=" << final_cost << endl;

    return final_cost;
}

void AntPlanHeuristic::initialize_python_function(const std::string &func_name) {
    py_func_name = func_name;
    ensure_python_ready();
}

void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready) return;
    py::initialize_interpreter();
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, ".");
    py::module mdl = py::module::import("antplan_model");
    py_cost_fn = mdl.attr(py_func_name.c_str());
    py_ready = true;
}

static shared_ptr<Heuristic> _parse(options::OptionParser &parser) {
    parser.document_synopsis("Anticipatory symbolic exploration heuristic (C++ + Python)", "");
    parser.document_property("admissible", "no");
    parser.document_property("consistent", "no");
    parser.document_property("safe", "yes");

    relaxation_heuristic::RelaxationHeuristic::add_options_to_parser(parser);
    parser.add_option<std::string>("function", "Python function to call", "anticipatory_cost_fn");

    options::Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;
    return make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

} // namespace antplan_heuristic
