#include "antplan_heuristic.h"
#include "relaxation_heuristic.h"
#include "../task_proxy.h"
#include "../utils/logging.h"
#include "../option_parser.h"
#include "../plugin.h"

#include <pybind11/stl.h>
#include <limits>
#include <iostream>
#include <algorithm>

using namespace std;
namespace py = pybind11;

namespace antplan_heuristic {

// Static Python integration
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name = "anticipatory_cost_fn";

AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : relaxation_heuristic::RelaxationHeuristic(opts) {
    std::string func_name = opts.get<std::string>("function");
    initialize_python_function(func_name);
    utils::g_log << "Initialized AntPlan heuristic with proper relaxed simulation." << endl;
}

void AntPlanHeuristic::setup_exploration_queue() {
    // Clear queue and reset proposition costs
    while (!queue.empty()) queue.pop();
    for (auto &prop : propositions)
        prop.cost = -1;

    for (auto &op : unary_operators) {
        op.unsatisfied_preconditions = op.num_preconditions;
        op.cost = op.base_cost;
    }
}

void AntPlanHeuristic::setup_exploration_queue_state(const State &state) {
    std::vector<int> facts;
    for (FactProxy fact : state)
        facts.push_back(fact.get_value());

    for (FactProxy fact : state) {
        PropID prop_id = get_prop_id(fact);
        enqueue_if_necessary(prop_id, 0, facts);
    }
}

void AntPlanHeuristic::enqueue_if_necessary(PropID prop_id, int cost, const std::vector<int> &facts) {
    Proposition *prop = get_proposition(prop_id);
    if (prop->cost == -1 || prop->cost > cost) {
        prop->cost = cost;
        queue.push({cost, prop_id, facts});
    }
}

int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);

    setup_exploration_queue();
    setup_exploration_queue_state(state);

    int unsolved_goals = goal_propositions.size();
    int best_combined_cost = std::numeric_limits<int>::max();

    cerr << "[ANTPLAN] Starting relaxed exploration with full simulation..." << endl;

    while (!queue.empty()) {
        RelaxedNode node = queue.top();
        queue.pop();

        int distance = node.cost;
        PropID prop_id = node.prop_id;
        Proposition *prop = get_proposition(prop_id);
        if (prop->cost < distance)
            continue;

        // Build snapshot for Python
        std::map<std::string, std::string> state_map;
        int num_vars = task_proxy.get_variables().size();
        for (int var_id = 0; var_id < num_vars; ++var_id) {
            VariableProxy var = task_proxy.get_variables()[var_id];
            FactProxy fact = var.get_fact(node.facts[var_id]);
            state_map[var.get_name()] = fact.get_name();
        }

        // Compute anticipatory cost
        int ant_cost = 0;
        try {
            ant_cost = py_cost_fn(py::cast(state_map)).cast<int>();
        } catch (const py::error_already_set &e) {
            cerr << "ANTPLAN: Python error:\n" << e.what() << endl;
            return DEAD_END;
        }

        int combined_cost = distance + ant_cost;
        best_combined_cost = min(best_combined_cost, combined_cost);

        // DEBUG
        cerr << "[ANTPLAN] Node expanded: dist=" << distance
             << " ant_cost=" << ant_cost
             << " combined=" << combined_cost << endl;

        if (prop->is_goal && --unsolved_goals == 0)
            break;

        // Expand successors
        for (auto op_id : precondition_of_pool.get_slice(prop->precondition_of, prop->num_precondition_occurences)) {
            UnaryOperator *unary_op = get_operator(op_id);
            unary_op->cost = max(unary_op->cost, unary_op->base_cost + prop->cost);
            --unary_op->unsatisfied_preconditions;

            if (unary_op->unsatisfied_preconditions == 0) {
                PropID effect_id = unary_op->effect;

                // Copy current facts and apply effect
                std::vector<int> new_facts = node.facts;
                int var_index = -1, value = -1;
                int current_offset = 0;
                for (int v = 0; v < num_vars; ++v) {
                    int domain_size = task_proxy.get_variables()[v].get_domain_size();
                    if (effect_id < current_offset + domain_size) {
                        var_index = v;
                        value = effect_id - current_offset;
                        break;
                    }
                    current_offset += domain_size;
                }
                if (var_index >= 0 && value >= 0)
                    new_facts[var_index] = value;

                enqueue_if_necessary(unary_op->effect, unary_op->cost, new_facts);
            }
        }
    }

    cerr << "[ANTPLAN] Best combined cost: " << best_combined_cost << endl;

    return (best_combined_cost == std::numeric_limits<int>::max())
               ? DEAD_END
               : best_combined_cost;
}

void AntPlanHeuristic::initialize_python_function(const std::string &func_name) {
    py_func_name = func_name;
    ensure_python_ready();
}

void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready)
        return;
    py::initialize_interpreter();
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(0, ".");
    py::module mdl = py::module::import("antplan_model");
    py_cost_fn = mdl.attr(py_func_name.c_str());
    py_ready = true;
}

static shared_ptr<Heuristic> _parse(options::OptionParser &parser) {
    parser.document_synopsis("Anticipatory heuristic with full relaxed simulation.", "");
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
