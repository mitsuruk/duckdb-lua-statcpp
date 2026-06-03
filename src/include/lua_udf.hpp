/**
 * @file lua_udf.hpp
 * @brief Bridge between DuckDB UDFs and Lua-based statistical functions
 *
 * Provides RegisterLuaStatcppFunctions(), which:
 *  1. Creates a Lua state with the statcpp C bindings pre-loaded (via lua_statcpp_bindings.hpp)
 *  2. Loads an external Lua script (e.g., stats.lua) that defines the user-facing functions
 *  3. Registers each Lua function as a DuckDB vectorized scalar UDF
 *
 * Registered SQL functions (prefixed with "lua_" to distinguish from direct C++ UDFs):
 *  - lua_stat_mad(LIST<DOUBLE>)            -> DOUBLE
 *  - lua_stat_hodges_lehmann(LIST<DOUBLE>) -> DOUBLE
 *  - lua_stat_missing_rate(LIST<DOUBLE>)   -> DOUBLE
 *  - lua_stat_impute_mean(LIST<DOUBLE>)    -> LIST<DOUBLE>
 *
 * PoC note: A single lua_State is shared across all UDFs (not thread-safe).
 * For production use, per-thread states or a mutex would be required.
 */

#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#include "duckdb.hpp"
#include "lua_statcpp_bindings.hpp"

namespace lua_duckdb {

// ---------------------------------------------------------------------------
// Type helpers (mirror of statcpp_udf.hpp for clarity)
// ---------------------------------------------------------------------------

inline duckdb::LogicalType ListOfDouble() {
    return duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE);
}

// ---------------------------------------------------------------------------
// DuckDB <-> Lua conversion helpers
// ---------------------------------------------------------------------------

/**
 * @brief Push a DuckDB LIST<DOUBLE> value onto the Lua stack as a 1-indexed table.
 *
 * NULL list elements are pushed as NaN (statcpp missing-value convention).
 * A NULL list itself results in an empty table being pushed.
 */
inline void PushListAsTable(lua_State* L, const duckdb::Value& list_val) {
    if (list_val.IsNull()) {
        lua_newtable(L);
        return;
    }
    const auto& children = duckdb::ListValue::GetChildren(list_val);
    lua_createtable(L, static_cast<int>(children.size()), 0);
    for (int i = 0; i < static_cast<int>(children.size()); ++i) {
        if (children[i].IsNull()) {
            lua_pushnumber(L, std::numeric_limits<double>::quiet_NaN());
        } else {
            lua_pushnumber(L, children[i].GetValue<double>());
        }
        lua_rawseti(L, -2, static_cast<lua_Integer>(i + 1));
    }
}

/**
 * @brief Convert a Lua table at stack index `idx` to a DuckDB LIST<DOUBLE> value.
 *
 * NaN entries are converted back to DuckDB NULL (SQL NULL).
 */
inline duckdb::Value TableToList(lua_State* L, int idx) {
    const int n = static_cast<int>(lua_rawlen(L, idx));
    std::vector<duckdb::Value> children;
    children.reserve(n);
    for (int i = 1; i <= n; ++i) {
        lua_rawgeti(L, idx, static_cast<lua_Integer>(i));
        if (lua_isnil(L, -1) || std::isnan(lua_tonumber(L, -1))) {
            children.push_back(duckdb::Value(duckdb::LogicalType::DOUBLE));
        } else {
            children.push_back(duckdb::Value::DOUBLE(lua_tonumber(L, -1)));
        }
        lua_pop(L, 1);
    }
    return duckdb::Value::LIST(duckdb::LogicalType::DOUBLE, std::move(children));
}

// ---------------------------------------------------------------------------
// Lua state wrapper
// ---------------------------------------------------------------------------

/**
 * @brief RAII wrapper around a lua_State.
 *
 * Initialises the state with standard libraries and the statcpp module,
 * then loads the given Lua script file.
 * Throws std::runtime_error if the script cannot be loaded or has syntax errors.
 */
struct LuaState {
    lua_State* L = nullptr;

    explicit LuaState(const std::string& script_path) {
        L = lua_statcpp::CreateLuaState();
        if (luaL_dofile(L, script_path.c_str()) != LUA_OK) {
            const std::string err = lua_tostring(L, -1);
            lua_close(L);
            L = nullptr;
            throw std::runtime_error("Failed to load Lua script '" + script_path + "': " + err);
        }
    }

    ~LuaState() {
        if (L != nullptr) {
            lua_close(L);
        }
    }

    LuaState(const LuaState&) = delete;
    LuaState& operator=(const LuaState&) = delete;
};

// ---------------------------------------------------------------------------
// UDF registration helpers
// ---------------------------------------------------------------------------

/**
 * @brief Register a DuckDB UDF that calls a Lua function: LIST<DOUBLE> -> DOUBLE
 *
 * @param con          DuckDB connection
 * @param state        Shared Lua state (must already have the script loaded)
 * @param sql_name     SQL function name to register (e.g., "lua_stat_mad")
 * @param lua_fn_name  Name of the global Lua function (e.g., "lua_mad")
 */
inline void RegisterLuaListToScalar(
    duckdb::Connection& con,
    std::shared_ptr<LuaState> state,
    const std::string& sql_name,
    const std::string& lua_fn_name
) {
    duckdb::scalar_function_t udf =
        [state, lua_fn_name](
            duckdb::DataChunk& args,
            duckdb::ExpressionState& /*es*/,
            duckdb::Vector& result
        ) {
        lua_State* L = state->L;
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            lua_getglobal(L, lua_fn_name.c_str());
            PushListAsTable(L, args.data[0].GetValue(i));
            if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
                lua_pop(L, 1);
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
                continue;
            }
            if (lua_isnil(L, -1)) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
            } else {
                result.SetValue(i, duckdb::Value::DOUBLE(lua_tonumber(L, -1)));
            }
            lua_pop(L, 1);
        }
    };
    con.CreateVectorizedFunction(sql_name, {ListOfDouble()}, duckdb::LogicalType::DOUBLE, udf);
}

/**
 * @brief Register a DuckDB UDF that calls a Lua function: LIST<DOUBLE> -> LIST<DOUBLE>
 *
 * @param con          DuckDB connection
 * @param state        Shared Lua state
 * @param sql_name     SQL function name to register
 * @param lua_fn_name  Name of the global Lua function
 */
inline void RegisterLuaListToList(
    duckdb::Connection& con,
    std::shared_ptr<LuaState> state,
    const std::string& sql_name,
    const std::string& lua_fn_name
) {
    duckdb::scalar_function_t udf =
        [state, lua_fn_name](
            duckdb::DataChunk& args,
            duckdb::ExpressionState& /*es*/,
            duckdb::Vector& result
        ) {
        lua_State* L = state->L;
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            lua_getglobal(L, lua_fn_name.c_str());
            PushListAsTable(L, args.data[0].GetValue(i));
            if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
                lua_pop(L, 1);
                result.SetValue(i, duckdb::Value(ListOfDouble()));
                continue;
            }
            if (!lua_istable(L, -1)) {
                result.SetValue(i, duckdb::Value(ListOfDouble()));
            } else {
                result.SetValue(i, TableToList(L, -1));
            }
            lua_pop(L, 1);
        }
    };
    con.CreateVectorizedFunction(sql_name, {ListOfDouble()}, ListOfDouble(), udf);
}

/**
 * @brief Register a DuckDB UDF that calls a Lua function: LIST<DOUBLE> -> VARCHAR
 *
 * Used for free-form report functions that build a string in Lua.
 *
 * @param con          DuckDB connection
 * @param state        Shared Lua state
 * @param sql_name     SQL function name to register
 * @param lua_fn_name  Name of the global Lua function
 */
inline void RegisterLuaListToString(
    duckdb::Connection& con,
    std::shared_ptr<LuaState> state,
    const std::string& sql_name,
    const std::string& lua_fn_name
) {
    duckdb::scalar_function_t udf =
        [state, lua_fn_name](
            duckdb::DataChunk& args,
            duckdb::ExpressionState& /*es*/,
            duckdb::Vector& result
        ) {
        lua_State* L = state->L;
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            lua_getglobal(L, lua_fn_name.c_str());
            PushListAsTable(L, args.data[0].GetValue(i));
            if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
                lua_pop(L, 1);
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::VARCHAR));
                continue;
            }
            if (lua_isnil(L, -1)) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::VARCHAR));
            } else {
                result.SetValue(i, duckdb::Value(std::string(lua_tostring(L, -1))));
            }
            lua_pop(L, 1);
        }
    };
    con.CreateVectorizedFunction(sql_name, {ListOfDouble()}, duckdb::LogicalType::VARCHAR, udf);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief Load a Lua script and register all Lua-backed statcpp UDFs.
 *
 * @param con          DuckDB connection
 * @param script_path  Absolute or relative path to the Lua script (e.g., stats.lua)
 *
 * Registered functions:
 *  - lua_stat_mad(LIST<DOUBLE>)            -> DOUBLE
 *  - lua_stat_hodges_lehmann(LIST<DOUBLE>) -> DOUBLE
 *  - lua_stat_missing_rate(LIST<DOUBLE>)   -> DOUBLE
 *  - lua_stat_impute_mean(LIST<DOUBLE>)    -> LIST<DOUBLE>
 *  - lua_stat_robust_cv(LIST<DOUBLE>)      -> DOUBLE        (composed in Lua)
 *  - lua_stat_smart_impute(LIST<DOUBLE>)   -> LIST<DOUBLE>  (policy in Lua)
 *  - lua_stat_summary(LIST<DOUBLE>)        -> VARCHAR       (free-form report)
 *
 * The last three are NOT statcpp functions: they are defined purely in
 * stats.lua (composition, policy and free-form reporting), demonstrating
 * extension without C++ changes beyond this one-time registration.
 */
inline void RegisterLuaStatcppFunctions(duckdb::Connection& con, const std::string& script_path) {
    auto state = std::make_shared<LuaState>(script_path);
    RegisterLuaListToScalar(con, state, "lua_stat_mad",            "lua_mad");
    RegisterLuaListToScalar(con, state, "lua_stat_hodges_lehmann", "lua_hodges_lehmann");
    RegisterLuaListToScalar(con, state, "lua_stat_missing_rate",   "lua_missing_rate");
    RegisterLuaListToList  (con, state, "lua_stat_impute_mean",    "lua_impute_mean");
    // Lua-only extensions (composition + policy + free-form report), no statcpp counterpart
    RegisterLuaListToScalar(con, state, "lua_stat_robust_cv",      "lua_robust_cv");
    RegisterLuaListToList  (con, state, "lua_stat_smart_impute",   "lua_smart_impute");
    RegisterLuaListToString(con, state, "lua_stat_summary",        "lua_summary");
}

}  // namespace lua_duckdb
