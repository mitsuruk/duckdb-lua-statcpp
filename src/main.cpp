/**
 * @file main.cpp
 * @brief Demo driver for the DuckDB + statcpp (+ Lua) statistical UDF layer
 *
 * Two registration paths are wired onto the same in-memory database:
 *   1. Direct C++ UDFs (statcpp_udf.hpp) — exposes ~50 statcpp functions to SQL.
 *   2. Lua-backed UDFs (lua_udf.hpp) — a Lua script (stats.lua) calls statcpp through
 *      a C binding, showing how new SQL functions can be authored in Lua with no C++
 *      recompilation.
 *
 * The program walks through each category of C++ function with a representative query,
 * confirms the Lua path matches the C++ path numerically, and then demonstrates the
 * Lua-only extensions (composition, policy and free-form reporting).
 *
 * LUA_STATS_SCRIPT_PATH is defined at compile time by cmake/lua.cmake and points to
 * src/lua/stats.lua in the project source tree.
 */

#include <iostream>
#include <stdexcept>
#include <string>

#include "duckdb.hpp"

#include "statcpp_udf.hpp"
#include "lua_udf.hpp"

namespace {

/// Run a query and print its result under a heading. On error, print the message.
void RunAndPrint(duckdb::Connection& con, const std::string& title, const std::string& sql) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << "SQL: " << sql << "\n";
    // result->Print() uses C stdio, so flush the C++ stream first to keep output ordered.
    std::cout << std::flush;
    auto result = con.Query(sql);
    if (result->HasError()) {
        std::cout << "ERROR: " << result->GetError() << "\n";
        return;
    }
    result->Print();
}

/// Populate the demo tables used throughout the walkthrough.
void CreateDemoData(duckdb::Connection& con) {
    // measurements: group A holds an outlier (1000.0); group B does not.
    con.Query("CREATE TABLE measurements (grp VARCHAR, v DOUBLE)");
    con.Query(
        "INSERT INTO measurements VALUES "
        "('A', 10.0), ('A', 11.0), ('A', 9.0), ('A', 10.5), ('A', 1000.0), "
        "('B', 20.0), ('B', 21.0), ('B', 19.0), ('B', 20.5), ('B', 22.0)");

    // sensor: a column with missing values (NULL) for the missing-data functions.
    con.Query("CREATE TABLE sensor (id INTEGER, reading DOUBLE)");
    con.Query(
        "INSERT INTO sensor VALUES "
        "(1, 10.0), (2, NULL), (3, 12.0), (4, 14.0), (5, NULL), (6, 16.0)");

    // paired: two correlated columns for the two-sample functions.
    con.Query("CREATE TABLE paired (x DOUBLE, y DOUBLE)");
    con.Query(
        "INSERT INTO paired VALUES "
        "(1.0, 2.1), (2.0, 3.9), (3.0, 6.2), (4.0, 7.8), (5.0, 10.1), (6.0, 12.2)");
}

}  // namespace

int main() {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

    // Register the direct C++ statcpp UDFs (table-driven, ~50 functions).
    statcpp_duckdb::RegisterStatcppFunctions(con);

    // Register Lua-backed UDFs (loads stats.lua at runtime).
    const std::string lua_script_path = LUA_STATS_SCRIPT_PATH;
    try {
        lua_duckdb::RegisterLuaStatcppFunctions(con, lua_script_path);
        std::cout << "Lua script loaded: " << lua_script_path << "\n";
    } catch (const std::runtime_error& e) {
        std::cout << "ERROR: " << e.what() << "\n";
        return 1;
    }

    CreateDemoData(con);

    // -----------------------------------------------------------------
    // 1. Descriptive statistics (location / spread / shape)
    // -----------------------------------------------------------------
    RunAndPrint(con, "descriptive: location, spread and shape per group",
                "SELECT grp, "
                "round(stat_mean(list(v)), 2) AS mean, "
                "round(stat_median(list(v)), 2) AS median, "
                "round(stat_stddev(list(v)), 2) AS stddev, "
                "round(stat_iqr(list(v)), 2) AS iqr, "
                "round(stat_skewness(list(v)), 3) AS skewness "
                "FROM measurements GROUP BY grp ORDER BY grp");

    // -----------------------------------------------------------------
    // 2. Order statistics with a parameter (percentile)
    // -----------------------------------------------------------------
    RunAndPrint(con, "order: min / median / 90th percentile / max per group",
                "SELECT grp, "
                "round(stat_minimum(list(v)), 2) AS min, "
                "round(stat_percentile(list(v), 0.5), 2) AS p50, "
                "round(stat_percentile(list(v), 0.9), 2) AS p90, "
                "round(stat_maximum(list(v)), 2) AS max "
                "FROM measurements GROUP BY grp ORDER BY grp");

    // -----------------------------------------------------------------
    // 3. Robust statistics (resistant to the outlier in group A)
    // -----------------------------------------------------------------
    RunAndPrint(con, "robust: non-robust (mean/stddev) vs robust (HL/MAD), note A's outlier",
                "SELECT grp, "
                "round(stat_mean(list(v)), 2) AS mean, "
                "round(stat_hodges_lehmann(list(v)), 2) AS hodges_lehmann, "
                "round(stat_stddev(list(v)), 2) AS stddev, "
                "round(stat_mad(list(v)), 2) AS mad, "
                "round(stat_trimmed_mean(list(v), 0.2), 2) AS trimmed_mean "
                "FROM measurements GROUP BY grp ORDER BY grp");

    // -----------------------------------------------------------------
    // 4. Two-sample statistics (correlation / covariance)
    // -----------------------------------------------------------------
    RunAndPrint(con, "two-sample: correlation and covariance of x and y",
                "SELECT "
                "round(stat_pearson_correlation(list(x), list(y)), 4) AS pearson, "
                "round(stat_spearman_correlation(list(x), list(y)), 4) AS spearman, "
                "round(stat_kendall_tau(list(x), list(y)), 4) AS kendall, "
                "round(stat_covariance(list(x), list(y)), 4) AS covariance "
                "FROM paired");

    // -----------------------------------------------------------------
    // 5. Column transforms (LIST -> LIST): ranking and winsorization
    // -----------------------------------------------------------------
    RunAndPrint(con, "transform: original vs rank vs winsorized (group A with outlier)",
                "WITH t AS ("
                "  SELECT unnest(list(v ORDER BY v)) AS original, "
                "         unnest(stat_rank_transform(list(v ORDER BY v))) AS rank, "
                "         unnest(stat_winsorize(list(v ORDER BY v))) AS winsorized "
                "  FROM measurements WHERE grp = 'A'"
                ") SELECT * FROM t");

    // -----------------------------------------------------------------
    // 6. Missing-data handling
    // -----------------------------------------------------------------
    RunAndPrint(con, "missing: rate, and per-row original vs mean/median imputation",
                "WITH cmp AS ("
                "  SELECT unnest(list(reading ORDER BY id)) AS original, "
                "         unnest(stat_impute_mean(list(reading ORDER BY id))) AS imp_mean, "
                "         unnest(stat_fillna_median(list(reading ORDER BY id))) AS imp_median "
                "  FROM sensor"
                ") SELECT *, "
                "(SELECT round(stat_missing_rate(list(reading)), 3) FROM sensor) AS missing_rate "
                "FROM cmp");

    // -----------------------------------------------------------------
    // 7. C++ path vs Lua path: results must match
    // -----------------------------------------------------------------
    // The same statistics are reachable through stats.lua (prefix "lua_").
    RunAndPrint(con, "parity: C++ stat_* vs Lua lua_stat_* (must match)",
                "SELECT grp, "
                "round(stat_hodges_lehmann(list(v)), 4) AS cpp_hl, "
                "round(lua_stat_hodges_lehmann(list(v)), 4) AS lua_hl, "
                "round(stat_mad(list(v)), 4) AS cpp_mad, "
                "round(lua_stat_mad(list(v)), 4) AS lua_mad, "
                "CASE WHEN stat_mad(list(v)) = lua_stat_mad(list(v)) THEN 'MATCH' ELSE 'MISMATCH' END AS check "
                "FROM measurements GROUP BY grp ORDER BY grp");

    // -----------------------------------------------------------------
    // 8. Lua-only extensions: new SQL functions authored purely in Lua
    // -----------------------------------------------------------------
    // robust_cv, smart_impute and summary have NO statcpp / C++ counterpart. They
    // are defined in src/lua/stats.lua on top of the bound statcpp primitives,
    // demonstrating that behaviour can be added by editing Lua, with no rebuild.
    std::cout << "\n--- Lua-only extensions (composition, policy, free-form report) ---\n"
                 "Authored in stats.lua on top of statcpp primitives; no C++ rebuild needed.\n";

    RunAndPrint(con, "lua-only: robust coefficient of variation per group",
                "SELECT grp, round(lua_stat_robust_cv(list(v)), 4) AS robust_cv "
                "FROM measurements GROUP BY grp ORDER BY grp");

    // sparse: 4 of 6 missing (rate 0.667 > 0.5) to trigger the policy's "refuse" branch.
    con.Query("CREATE TABLE sparse (id INTEGER, reading DOUBLE)");
    con.Query(
        "INSERT INTO sparse VALUES "
        "(1, 10.0), (2, NULL), (3, NULL), (4, NULL), (5, NULL), (6, 16.0)");

    RunAndPrint(con, "lua-only: smart_impute policy (sensor 33% missing -> imputes)",
                "SELECT round(lua_stat_missing_rate(list(reading)), 3) AS missing_rate, "
                "lua_stat_smart_impute(list(reading ORDER BY id)) AS result "
                "FROM sensor");

    RunAndPrint(con, "lua-only: smart_impute policy (sparse 67% missing -> refuses)",
                "SELECT round(lua_stat_missing_rate(list(reading)), 3) AS missing_rate, "
                "lua_stat_smart_impute(list(reading ORDER BY id)) AS result, "
                "lua_stat_smart_impute(list(reading ORDER BY id)) IS NULL AS refused "
                "FROM sparse");

    RunAndPrint(con, "lua-only: free-form summary report per group",
                "SELECT grp, lua_stat_summary(list(v)) AS summary "
                "FROM measurements GROUP BY grp ORDER BY grp");

    std::cout << "\nDone.\n";
    return 0;
}
