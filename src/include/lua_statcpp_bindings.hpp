/**
 * @file lua_statcpp_bindings.hpp
 * @brief Lua C bindings that expose statcpp functions as a Lua module named "statcpp"
 *
 * Usage in Lua scripts:
 *   local statcpp = require("statcpp")
 *   local result  = statcpp.mad(data_table)
 *
 * The module must be pre-loaded into the lua_State before executing any script
 * that calls require("statcpp"). Use CreateLuaState() to get a ready-to-use state.
 *
 * All functions accept a Lua table of numbers (1-indexed array).
 * nil entries in the table are treated as missing values (NaN) and are
 * excluded from statistics that cannot handle them.
 * Errors from statcpp (e.g., empty input) are caught and returned as nil.
 *
 * Thread safety: a lua_State is NOT thread-safe. Do not share a single state
 * across threads. For PoC single-threaded use this is acceptable.
 */

#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#include "statcpp/statcpp.hpp"

namespace lua_statcpp {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * @brief Convert a Lua table at stack index `idx` to std::vector<double>.
 *
 * nil entries become NaN (statcpp missing-value convention).
 * Returns an empty vector if the value at `idx` is not a table.
 */
inline std::vector<double> TableToVector(lua_State* L, int idx) {
    std::vector<double> result;
    if (!lua_istable(L, idx)) {
        return result;
    }
    const int n = static_cast<int>(lua_rawlen(L, idx));
    result.reserve(n);
    for (int i = 1; i <= n; ++i) {
        lua_rawgeti(L, idx, static_cast<lua_Integer>(i));
        if (lua_isnil(L, -1)) {
            result.push_back(std::numeric_limits<double>::quiet_NaN());
        } else {
            result.push_back(lua_tonumber(L, -1));
        }
        lua_pop(L, 1);
    }
    return result;
}

/**
 * @brief Push a std::vector<double> onto the Lua stack as a new table.
 *
 * NaN values are pushed as nil (matching statcpp missing-value convention).
 */
inline void VectorToTable(lua_State* L, const std::vector<double>& v) {
    lua_createtable(L, static_cast<int>(v.size()), 0);
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (std::isnan(v[i])) {
            lua_pushnil(L);
        } else {
            lua_pushnumber(L, v[i]);
        }
        lua_rawseti(L, -2, static_cast<lua_Integer>(i + 1));
    }
}

// ---------------------------------------------------------------------------
// Lua C functions (statcpp module entries)
// ---------------------------------------------------------------------------

/**
 * @brief statcpp.mad(table) -> number | nil
 *
 * Median Absolute Deviation (robust scale estimator, unscaled).
 * Missing values (nil/NaN) are excluded before calling statcpp::mad.
 */
static int LuaStatcppMad(lua_State* L) {
    const auto v = TableToVector(L, 1);
    std::vector<double> clean;
    clean.reserve(v.size());
    for (const double x : v) {
        if (!statcpp::is_na(x)) {
            clean.push_back(x);
        }
    }
    try {
        lua_pushnumber(L, statcpp::mad(clean.begin(), clean.end()));
    } catch (...) {
        lua_pushnil(L);
    }
    return 1;
}

/**
 * @brief statcpp.hodges_lehmann(table) -> number | nil
 *
 * Hodges-Lehmann location estimator (robust location estimator).
 * Missing values are excluded.
 */
static int LuaStatcppHodgesLehmann(lua_State* L) {
    const auto v = TableToVector(L, 1);
    std::vector<double> clean;
    clean.reserve(v.size());
    for (const double x : v) {
        if (!statcpp::is_na(x)) {
            clean.push_back(x);
        }
    }
    try {
        lua_pushnumber(L, statcpp::hodges_lehmann(clean.begin(), clean.end()));
    } catch (...) {
        lua_pushnil(L);
    }
    return 1;
}

/**
 * @brief statcpp.missing_rate(table) -> number | nil
 *
 * Fraction of nil/NaN entries in the input (uses statcpp::analyze_missing_patterns).
 */
static int LuaStatcppMissingRate(lua_State* L) {
    const auto v = TableToVector(L, 1);
    try {
        std::vector<std::vector<double>> matrix;
        matrix.reserve(v.size());
        for (const double x : v) {
            matrix.push_back({x});
        }
        const auto info = statcpp::analyze_missing_patterns(matrix);
        lua_pushnumber(L, info.missing_rates.at(0));
    } catch (...) {
        lua_pushnil(L);
    }
    return 1;
}

/**
 * @brief statcpp.impute_mean(table) -> table | nil
 *
 * Replaces each nil/NaN entry with the mean of the observed values.
 * Returns a new Lua table of the same length.
 */
static int LuaStatcppImputeMean(lua_State* L) {
    const auto v = TableToVector(L, 1);
    try {
        std::vector<double> observed;
        observed.reserve(v.size());
        for (const double x : v) {
            if (!statcpp::is_na(x)) {
                observed.push_back(x);
            }
        }
        const double fill = observed.empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : statcpp::mean(observed.begin(), observed.end());

        std::vector<double> out;
        out.reserve(v.size());
        for (const double x : v) {
            out.push_back(statcpp::is_na(x) ? fill : x);
        }
        VectorToTable(L, out);
    } catch (...) {
        lua_pushnil(L);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

static const luaL_Reg kStatcppFunctions[] = {
    {"mad",             LuaStatcppMad},
    {"hodges_lehmann",  LuaStatcppHodgesLehmann},
    {"missing_rate",    LuaStatcppMissingRate},
    {"impute_mean",     LuaStatcppImputeMean},
    {nullptr, nullptr}
};

/**
 * @brief Lua module opener for "statcpp".
 *
 * Called by luaL_requiref. Pushes the module table onto the stack.
 */
inline int luaopen_statcpp(lua_State* L) {
    luaL_newlib(L, kStatcppFunctions);
    return 1;
}

/**
 * @brief Create a new Lua state with standard libraries and the statcpp module pre-loaded.
 *
 * After this call, require("statcpp") works in any script executed on the returned state.
 * The caller owns the returned lua_State and must call lua_close() when done.
 */
inline lua_State* CreateLuaState() {
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    // Pre-register the statcpp module so require("statcpp") resolves without a file search
    luaL_requiref(L, "statcpp", luaopen_statcpp, 0);
    lua_pop(L, 1);  // pop the module table pushed by luaL_requiref
    return L;
}

}  // namespace lua_statcpp
