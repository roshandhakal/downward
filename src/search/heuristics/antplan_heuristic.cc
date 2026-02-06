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

namespace antplan_heuristic {

// ===== Static definitions =====
py::object AntPlanHeuristic::py_cost_fn;
bool AntPlanHeuristic::py_ready = false;
std::string AntPlanHeuristic::py_func_name  = "anticipatory_cost_fn";
std::string AntPlanHeuristic::py_module_name = "antplan.scripts.eval_antplan_gripper";

// ---- Fast-path tables (built once) ----
static bool g_py_tables_ready = false;
static vector<py::str> g_py_var_names;                  // [var_id]
static vector<vector<py::str>> g_py_fact_names;         // [var_id][value]

// ---- Options (set from parser) ----
static bool g_debug = false;        // print occasional diagnostics
static bool g_log_states = false;   // VERY slow if true
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
    // FNV-1a 64-bit
    h ^= x;
    h *= 1099511628211ULL;
    return h;
}

static uint64_t hash_state_values(const TaskProxy &task_proxy, const State &state) {
    uint64_t h = 1469598103934665603ULL;
    int num_vars = task_proxy.get_variables().size();
    for (int var_id = 0; var_id < num_vars; ++var_id) {
        // FactProxy supports get_value() in FD
        FactProxy fact = state[var_id];
        uint64_t v = static_cast<uint64_t>(fact.get_value());
        h = fnv1a_64_update(h, v + 0x9e3779b97f4a7c15ULL + (static_cast<uint64_t>(var_id) << 1));
    }
    return h;
}

static void maybe_evict_cache() {
    if (!g_use_cache) return;
    if (g_cache.size() <= g_cache_max_entries) return;

    // simple eviction: clear (fast + predictable)
    // if you want LRU, implement it, but this is often enough.
    g_cache.clear();
}

static void ensure_py_tables_ready(TaskProxy &task_proxy) {
    if (g_py_tables_ready)
        return;

    // Needs Python initialized and GIL held
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
    // Handle torch scalars / numpy scalars: use .item() if present
    if (py::hasattr(obj, "item")) {
        obj = obj.attr("item")();
    }
    return obj.cast<double>();
}


// ===== ctor / dtor =====
AntPlanHeuristic::AntPlanHeuristic(const options::Options &opts)
    : AdditiveHeuristic(opts),
      relaxed_plan(task_proxy.get_operators().size(), false) {

    // Only two required options: function + module
    std::string func_name = opts.get<std::string>("function");
    std::string mod_name  = opts.get<std::string>("module");

    py_func_name   = func_name;
    py_module_name = mod_name;
    py_ready       = false;

    utils::g_log << "[AntPlan] ctor: function=" << py_func_name
                 << " module=" << (py_module_name.empty() ? "<none>" : py_module_name)
                 << std::endl;

    ensure_python_ready();

    // Build tables once interpreter is ready
    ensure_py_tables_ready(task_proxy);
}

AntPlanHeuristic::~AntPlanHeuristic() {
    // Do not finalize interpreter here.
}

// ===== Python init =====
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
        if (py_module_name.empty())
            throw std::runtime_error("No Python module provided for AntPlan.");

        py::module sys = py::module::import("sys");
        // Insert "." only if not already present (avoid growing sys.path)
        py::list p = sys.attr("path");
        bool has_dot = false;
        for (auto item : p) {
            if (py::cast<std::string>(item) == ".") { has_dot = true; break; }
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
        // Fail fast: if Python isn't ready, heuristic is useless and will waste time
        utils::g_log << "[AntPlan] Failed to initialize Python: " << e.what() << std::endl;
        throw;
    }
}

// ===== preferred operators helper (unchanged) =====
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
    ++g_calls;

    // Convert ancestor state as FD expects
    State state = convert_ancestor_state(ancestor_state);

    // Optional cache: huge win if states repeat (reopenings/duplicates/multi-evals)
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
        // Fail fast (don’t silently return 0 and pretend it’s fine)
        throw std::runtime_error("[AntPlan] Python not ready in compute_heuristic.");
    }

    // Build Python dict without std::map allocations
    int num_vars = task_proxy.get_variables().size();

    int anticipatory_cost_int = 0;

    try {
        py::gil_scoped_acquire gil;

        // Tables should exist; if task changes, you may need to rebuild.
        ensure_py_tables_ready(task_proxy);

        py::dict d;
        d.attr("clear")();

        for (int var_id = 0; var_id < num_vars; ++var_id) {
            FactProxy fact = state[var_id];
            int val = fact.get_value();
            // var_id and val are valid indices
            d[g_py_var_names[var_id]] = g_py_fact_names[var_id][val];
        }

        py::object res = py_cost_fn(d);

        double anticipatory_cost = py_scalar_to_double(res);

        // Convert to int for FD (you can change this to keep double if you edit other pieces)
        if (std::isnan(anticipatory_cost) || std::isinf(anticipatory_cost)) {
            anticipatory_cost_int = 0;
        } else {
            // Round to nearest integer (or floor/ceil depending on what you want)
            anticipatory_cost_int = static_cast<int>(std::lround(anticipatory_cost));
            if (anticipatory_cost_int < 0) anticipatory_cost_int = 0;
        }

    } catch (const std::exception &e) {
        // Log once-ish (optional)
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
        // Make failure obvious to the search (or choose a penalty)
        anticipatory_cost_int = 0;
    }

    // DO NOT print per-state facts by default (this is extremely slow)
    if (g_log_states) {
        utils::g_log << "[AntPlan] State facts:\n";
        for (int var_id = 0; var_id < num_vars; ++var_id) {
            VariableProxy var = task_proxy.get_variables()[var_id];
            FactProxy fact = state[var_id];
            utils::g_log << "  " << var.get_name() << " = " << fact.get_name() << "\n";
        }
        utils::g_log << "[AntPlan] anticipatory_cost_int=" << anticipatory_cost_int << "\n";
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
        "Evaluates a state with a Python cost function (anticipatory cost).");

    parser.add_option<std::string>(
        "function",
        "Python function name to call (attribute in module).",
        "anticipatory_cost_fn");

    parser.add_option<std::string>(
        "module",
        "Python module name to import (e.g., 'pkg.subpkg.module').",
        "antplan.scripts.eval_antplan_gripper");

    // ---- performance/debug knobs ----
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

    Heuristic::add_options_to_parser(parser);
    Options opts = parser.parse();
    if (parser.dry_run())
        return nullptr;

    // Bind options into file-scope flags (no header changes needed)
    g_debug = opts.get<bool>("debug");
    g_log_states = opts.get<bool>("log_states");
    g_use_cache = opts.get<bool>("cache");
    g_cache_max_entries = static_cast<size_t>(std::max(0, opts.get<int>("cache_max_entries")));

    return std::make_shared<AntPlanHeuristic>(opts);
}

static Plugin<Evaluator> _plugin("antplan", _parse);

} // namespace antplan_heuristic
