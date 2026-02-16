#include "antplan_heuristic.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../task_proxy.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;
using namespace relaxation_heuristic;

namespace antplan_heuristic {

// ===== Static definitions =====
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name = "anticipatory_cost_fn";
std::string AntPlanHeuristic::py_module_name = "antplan.scripts.eval_antplan_gripper";

int AntPlanHeuristic::evaluation_count = 0;
int AntPlanHeuristic::exploration_count = 0;
std::unordered_set<uint64_t> AntPlanHeuristic::explored_states;

// ---- Fast-path tables (built once) ----
static bool g_py_tables_ready = false;
static vector<py::str> g_py_var_names;                  // [var_id]
static vector<vector<py::str>> g_py_fact_names;         // [var_id][value]

// ---- Options (set from parser) ----
static bool g_debug = false;
static bool g_log_states = false;
static bool g_use_cache = true;
static size_t g_cache_max_entries = 500000;

// ---- Simple memo cache (keyed by 64-bit hash of state values) ----
static unordered_map<uint64_t, int> g_cache;

// ---- Stats ----
static uint64_t g_calls = 0;
static uint64_t g_cache_hits = 0;
static uint64_t g_cache_misses = 0;

// ===== helpers =====
static inline uint64_t fnv1a_64_update(uint64_t h, uint64_t x) {
    h ^= x;
    h *= 1099511628211ULL;
    return h;
}

static uint64_t hash_state_values(const TaskProxy &task_proxy, const State &state) {
    uint64_t h = 1469598103934665603ULL;
    int num_vars = task_proxy.get_variables().size();
    for (int var_id = 0; var_id < num_vars; ++var_id) {
        FactProxy fact = state[var_id];
        uint64_t v = static_cast<uint64_t>(fact.get_value());
        h = fnv1a_64_update(h, v + 0x9e3779b97f4a7c15ULL + (static_cast<uint64_t>(var_id) << 1));
    }
    return h;
}

static void maybe_evict_cache() {
    if (!g_use_cache) return;
    if (g_cache.size() <= g_cache_max_entries) return;
    g_cache.clear();
}

static void ensure_py_tables_ready(TaskProxy &task_proxy) {
    if (g_py_tables_ready)
        return;

    py::gil_scoped_acquire gil;

    int num_vars = task_proxy.get_variables().size();
    g_py_var_names.clear();
    g_py_fact_names.clear();
    g_py_var_names.reserve(num_vars);
    g_py_fact_names.resize(num_vars);

    for (int var_id = 0; var_id < num_vars; ++var_id) {
        VariableProxy var = task_proxy.get_variables()[var_id];
        g_py_var_names.emplace_back(py::str(var.get_name()));

        int dom = var.get_domain_size();
        g_py_fact_names[var_id].reserve(dom);
        for (int val = 0; val < dom; ++val) {
            FactProxy f = var.get_fact(val);
            g_py_fact_names[var_id].emplace_back(py::str(f.get_name()));
        }
    }

    g_py_tables_ready = true;

    if (g_debug) {
        utils::g_log << "[AntPlan] Built Python string tables for "
                     << num_vars << " variables.\n";
    }
}

static inline double py_scalar_to_double(py::object obj) {
    if (py::hasattr(obj, "item")) {
        obj = obj.attr("item")();
    }
    return obj.cast<double>();
}

// ===== ctor / dtor =====
AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : AdditiveHeuristic(opts),
      exploration_frequency(10),
      exploration_depth(2),
      improvement_threshold(0.9),
      exploration_budget(20),
      relaxed_plan(task_proxy.get_operators().size(), false) {

    std::string func_name = opts.get<std::string>("function");
    std::string mod_name = opts.get<std::string>("module");

    py_func_name = func_name;
    py_module_name = mod_name;
    py_ready = false;

    // Get exploration parameters
    exploration_frequency = opts.get<int>("exploration_frequency");
    exploration_depth = opts.get<int>("exploration_depth");
    improvement_threshold = opts.get<double>("improvement_threshold");
    exploration_budget = opts.get<int>("exploration_budget");

    utils::g_log << "[AntPlan] ctor: function=" << py_func_name
                 << " module=" << (py_module_name.empty() ? "<none>" : py_module_name)
                 << "\n[AntPlan] Exploration: freq=" << exploration_frequency
                 << " depth=" << exploration_depth
                 << " threshold=" << improvement_threshold
                 << " budget=" << exploration_budget
                 << std::endl;

    ensure_python_ready();
    ensure_py_tables_ready(task_proxy);
}

AntPlanHeuristic::~AntPlanHeuristic() {
    if (g_debug) {
        utils::g_log << "[AntPlan] Stats: "
                     << "total_calls=" << g_calls
                     << " cache_hits=" << g_cache_hits
                     << " cache_misses=" << g_cache_misses
                     << " explorations=" << exploration_count
                     << std::endl;
    }
}

// ===== Python init =====
void AntPlanHeuristic::ensure_python_ready() {
    if (py_ready)
        return;

    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    py::gil_scoped_acquire gil;

    try {
        if (py_module_name.empty())
            throw std::runtime_error("No Python module provided for AntPlan.");

        py::module sys = py::module::import("sys");
        py::list p = sys.attr("path");
        bool has_dot = false;
        for (auto item : p) {
            if (py::cast<std::string>(item) == ".") {
                has_dot = true;
                break;
            }
        }
        if (!has_dot) {
            sys.attr("path").attr("insert")(0, ".");
        }

        py::object mdl = py::module::import(py_module_name.c_str());

        if (!py::hasattr(mdl, py_func_name.c_str())) {
            throw std::runtime_error(
                "Python object '" + py_func_name + "' not found in module '" + py_module_name + "'");
        }

        py_cost_fn = mdl.attr(py_func_name.c_str());
        py_ready = true;

        if (g_debug) {
            utils::g_log << "[AntPlan] Python ready.\n";
        }
    } catch (const std::exception &e) {
        py_ready = false;
        utils::g_log << "[AntPlan] Failed to initialize Python: " << e.what() << std::endl;
        throw;
    }
}

// ===== preferred operators helper =====
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

// ===== Exploration methods =====
bool AntPlanHeuristic::should_explore_now() {
    if (evaluation_count < 100) {
        return (evaluation_count % 5 == 0);
    } else if (evaluation_count < 500) {
        return (evaluation_count % exploration_frequency == 0);
    } else {
        return (evaluation_count % (exploration_frequency * 2) == 0);
    }
}

double AntPlanHeuristic::evaluate_state_with_nn(const State &state) {
    py::gil_scoped_acquire gil;

    int num_vars = task_proxy.get_variables().size();
    py::dict d;

    for (int var_id = 0; var_id < num_vars; ++var_id) {
        FactProxy fact = state[var_id];
        d[g_py_var_names[var_id]] = g_py_fact_names[var_id][fact.get_value()];
    }

    py::object res = py_cost_fn(d);
    return py_scalar_to_double(res);
}

void AntPlanHeuristic::probe_successors(const State &state, int current_cost,
                                        int depth, int &budget) {
    if (depth == 0 || budget <= 0) return;

    uint64_t state_hash = hash_state_values(task_proxy, state);
    if (explored_states.count(state_hash)) return;
    explored_states.insert(state_hash);

    // Periodically clear explored_states to avoid memory growth
    if (explored_states.size() > 10000) {
        explored_states.clear();
    }

    py::gil_scoped_acquire gil;

    std::vector<std::pair<OperatorProxy, double>> promising_ops;

    for (OperatorProxy op : task_proxy.get_operators()) {
        if (!task_properties::is_applicable(op, state)) continue;
        if (--budget < 0) break;

        State succ = state.get_unregistered_successor(op);
        double succ_cost = evaluate_state_with_nn(succ);

        if (succ_cost < current_cost * improvement_threshold) {
            promising_ops.push_back({op, succ_cost});
        }
    }

    // Sort by cost (best first) - explicit types instead of auto
    std::sort(promising_ops.begin(), promising_ops.end(),
              [](const std::pair<OperatorProxy, double> &a, 
                 const std::pair<OperatorProxy, double> &b) { 
                  return a.second < b.second; 
              });

    for (size_t i = 0; i < std::min(size_t(3), promising_ops.size()); ++i) {
        set_preferred(promising_ops[i].first);

        if (g_debug) {
            utils::g_log << "[AntPlan] Depth " << (exploration_depth - depth)
                        << ": Preferring " << promising_ops[i].first.get_name()
                        << " (cost: " << current_cost << " -> " << promising_ops[i].second << ")\n";
        }
    }

    for (size_t i = 0; i < std::min(size_t(2), promising_ops.size()); ++i) {
        if (budget <= 0) break;

        State succ = state.get_unregistered_successor(promising_ops[i].first);
        probe_successors(succ, static_cast<int>(promising_ops[i].second), depth - 1, budget);
    }
}

void AntPlanHeuristic::explore_from_state(const State &state, int current_cost) {
    ++exploration_count;

    int budget = exploration_budget;

    if (g_debug) {
        utils::g_log << "[AntPlan] === Exploration #" << exploration_count
                    << " at eval " << evaluation_count
                    << " (budget: " << budget << ", depth: " << exploration_depth << ") ===\n";
    }

    probe_successors(state, current_cost, exploration_depth, budget);

    if (g_debug) {
        utils::g_log << "[AntPlan] Exploration used " << (exploration_budget - budget)
                    << "/" << exploration_budget << " budget\n";
    }
}

// ===== main computation =====
int AntPlanHeuristic::compute_heuristic(const State &ancestor_state) {
    ++g_calls;
    ++evaluation_count;

    State state = convert_ancestor_state(ancestor_state);

    if (g_use_cache) {
        uint64_t key = hash_state_values(task_proxy, state);
        auto it = g_cache.find(key);
        if (it != g_cache.end()) {
            ++g_cache_hits;
            return it->second;
        }
        ++g_cache_misses;
    }

    if (!py_ready) {
        throw std::runtime_error("[AntPlan] Python not ready in compute_heuristic.");
    }

    int anticipatory_cost_int = 0;

    try {
        py::gil_scoped_acquire gil;
        ensure_py_tables_ready(task_proxy);

        py::dict d;
        int num_vars = task_proxy.get_variables().size();
        for (int var_id = 0; var_id < num_vars; ++var_id) {
            FactProxy fact = state[var_id];
            d[g_py_var_names[var_id]] = g_py_fact_names[var_id][fact.get_value()];
        }

        py::object res = py_cost_fn(d);
        double anticipatory_cost = py_scalar_to_double(res);

        if (std::isnan(anticipatory_cost) || std::isinf(anticipatory_cost)) {
            anticipatory_cost_int = 0;
        } else {
            anticipatory_cost_int = static_cast<int>(std::lround(anticipatory_cost));
            if (anticipatory_cost_int < 0) anticipatory_cost_int = 0;
        }

    } catch (const std::exception &e) {
        if (g_debug) {
            try {
                py::gil_scoped_acquire gil2;
                py::object tb = py::module::import("traceback").attr("format_exc")();
                utils::g_log << "[AntPlan] Python function failed: " << e.what() << "\n"
                             << py::cast<std::string>(tb) << std::endl;
            } catch (...) {
                utils::g_log << "[AntPlan] Python function failed: " << e.what() << std::endl;
            }
        }
        anticipatory_cost_int = 0;
    }

    if (g_log_states) {
        utils::g_log << "[AntPlan] State facts:\n";
        int num_vars = task_proxy.get_variables().size();
        for (int var_id = 0; var_id < num_vars; ++var_id) {
            VariableProxy var = task_proxy.get_variables()[var_id];
            FactProxy fact = state[var_id];
            utils::g_log << "  " << var.get_name() << " = " << fact.get_name() << "\n";
        }
        utils::g_log << "[AntPlan] anticipatory_cost_int=" << anticipatory_cost_int << "\n";
    }

    // === EXPLORATION ===
    if (should_explore_now()) {
        explore_from_state(state, anticipatory_cost_int);
    }

    if (g_use_cache) {
        uint64_t key = hash_state_values(task_proxy, state);
        maybe_evict_cache();
        g_cache[key] = anticipatory_cost_int;
    }

    return anticipatory_cost_int;
}

// ===== plugin registration =====
static std::shared_ptr<Heuristic> _parse(OptionParser &parser) {
    parser.document_synopsis(
        "AntPlan heuristic",
        "Evaluates a state with a Python cost function (anticipatory cost) and explores promising branches.");

    parser.add_option<std::string>(
        "function",
        "Python function name to call (attribute in module).",
        "anticipatory_cost_fn");

    parser.add_option<std::string>(
        "module",
        "Python module name to import (e.g., 'pkg.subpkg.module').",
        "antplan.scripts.eval_antplan_gripper");

    // Performance/debug knobs
    parser.add_option<bool>(
        "debug",
        "Print tracebacks/diagnostics on Python failure (slow).",
        "false");
    parser.add_option<bool>(
        "log_states",
        "Log full state facts for every heuristic call (VERY slow).",
        "false");
    parser.add_option<bool>(
        "cache",
        "Memoize heuristic values by state hash.",
        "true");
    parser.add_option<int>(
        "cache_max_entries",
        "Max entries before cache is cleared (simple eviction).",
        "500000");

    // Exploration parameters
    parser.add_option<int>(
        "exploration_frequency",
        "Explore every N state evaluations (lower = more exploration).",
        "10");
    parser.add_option<int>(
        "exploration_depth",
        "How many actions to look ahead during exploration.",
        "2");
    parser.add_option<double>(
        "improvement_threshold",
        "State is 'good' if cost < current * threshold (0.9 = 10% improvement).",
        "0.9");
    parser.add_option<int>(
        "exploration_budget",
        "Max successor evaluations per exploration (prevents explosion).",
        "20");

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;

    g_debug = opts.get<bool>("debug");
    g_log_states = opts.get<bool>("log_states");
    g_use_cache = opts.get<bool>("cache");
    g_cache_max_entries = static_cast<size_t>(std::max(0, opts.get<int>("cache_max_entries")));

    return std::make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

} // namespace antplan_heuristic