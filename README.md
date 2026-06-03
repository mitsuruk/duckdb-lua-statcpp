# duckdb-lua-statcpp

[![CI](https://github.com/mitsuruk/duckdb-lua-statcpp/actions/workflows/ci.yml/badge.svg)](https://github.com/mitsuruk/duckdb-lua-statcpp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![CMake](https://img.shields.io/badge/build-CMake-064F8C?logo=cmake)

Statistical SQL functions for **DuckDB**, backed by the C++17
[statcpp](https://github.com/mitsuruk/statcpp) library, with an optional **Lua**
layer for authoring new functions without recompiling C++.

Including a couple of headers and calling one registration function gives a
DuckDB connection ~50 statistical SQL functions — descriptive statistics, order
statistics, robust estimators, correlation/covariance, data transforms and
missing-data handling — all computed by statcpp. The same data path is also
reachable through Lua, so new SQL functions can be composed in a script and take
effect on the next run with no rebuild.

Released under the [MIT License](LICENSE.md). DuckDB, statcpp and Lua are
third-party dependencies under their own licenses — see [License](#license).

## What you get

Two header-only layers you register on a `duckdb::Connection`:

```cpp
#include "statcpp_udf.hpp"   // direct C++ path: ~50 statcpp functions as SQL UDFs
#include "lua_udf.hpp"       // optional Lua path: author new functions in Lua

duckdb::DuckDB db(nullptr);
duckdb::Connection con(db);

statcpp_duckdb::RegisterStatcppFunctions(con);                       // stat_*
lua_duckdb::RegisterLuaStatcppFunctions(con, "src/lua/stats.lua");   // lua_stat_*
```

Functions operate on a `LIST<DOUBLE>`, so a column is aggregated with the
built-in `list()` aggregate first:

```sql
-- Descriptive + robust statistics per group
SELECT grp,
       stat_mean(list(v))            AS mean,
       stat_median(list(v))          AS median,
       stat_stddev(list(v))          AS stddev,
       stat_hodges_lehmann(list(v))  AS robust_location,
       stat_mad(list(v))             AS robust_scale,
       stat_percentile(list(v), 0.9) AS p90
FROM measurements
GROUP BY grp;

-- Correlation between two columns
SELECT stat_pearson_correlation(list(x), list(y)) AS r FROM paired;

-- Column transform: mean-impute missing values, preserving row order
SELECT unnest(stat_impute_mean(list(reading ORDER BY id))) AS imputed FROM sensor;
```

> **Statistics reference.** This repository only *exposes* statcpp to SQL. For
> the definition, assumptions, parameters and algorithm of each statistic, see
> the [statcpp documentation](https://github.com/mitsuruk/statcpp). Names below
> mirror statcpp with a `stat_` prefix.

## Function catalog

All functions take one or two `LIST<DOUBLE>` arguments. Missing values
(SQL `NULL` / `NaN`) are dropped before single- and two-sample statistics
(pairwise for two-sample); on invalid input (e.g. empty) the function returns
SQL `NULL`.

### Single sample → scalar — `(LIST<DOUBLE>) → DOUBLE`

| Category | Functions |
| --- | --- |
| Central tendency | `stat_mean` `stat_median` `stat_mode` `stat_geometric_mean` `stat_harmonic_mean` |
| Totals / counts | `stat_sum` `stat_count` `stat_minimum` `stat_maximum` |
| Dispersion | `stat_range` `stat_variance` `stat_population_variance` `stat_sample_variance` `stat_stddev` `stat_population_stddev` `stat_sample_stddev` `stat_coefficient_of_variation` `stat_iqr` |
| Shape | `stat_skewness` `stat_sample_skewness` `stat_population_skewness` `stat_kurtosis` `stat_sample_kurtosis` `stat_population_kurtosis` |
| Robust | `stat_mad` `stat_mad_scaled` `stat_hodges_lehmann` `stat_biweight_midvariance` |

### Parameterized → scalar — `(LIST<DOUBLE>, DOUBLE) → DOUBLE`

| Function | Parameter |
| --- | --- |
| `stat_percentile` | proportion `p` in `[0, 1]` (e.g. `0.9` = 90th percentile) |
| `stat_trimmed_mean` | proportion trimmed per side, in `[0, 0.5)` |

### Two samples → scalar — `(LIST<DOUBLE>, LIST<DOUBLE>) → DOUBLE`

`stat_covariance` `stat_sample_covariance` `stat_population_covariance`
`stat_pearson_correlation` `stat_spearman_correlation` `stat_kendall_tau`

### Column transforms — `(LIST<DOUBLE>) → LIST<DOUBLE>`

| Category | Functions |
| --- | --- |
| Missing-value fill | `stat_impute_mean` `stat_fillna_mean` `stat_fillna_median` `stat_fillna_ffill` `stat_fillna_bfill` `stat_fillna_interpolate` |
| Math transforms | `stat_log_transform` `stat_sqrt_transform` `stat_rank_transform` `stat_winsorize` |
| Missing rate | `stat_missing_rate` *(→ scalar)* |

Wrap a transform result in `unnest()` to expand it back into rows; keep input
order with `list(col ORDER BY key)`.

### Scope

Functions that fit a SQL scalar/list signature are exposed. statcpp routines that
return rich objects (regression and GLM models, ANOVA tables, hypothesis-test
result structures, distribution objects, clustering, survival models, …) are out
of scope here — call statcpp directly in C++ for those.

## Extending in Lua (sample)

The same statistics are reachable through Lua-backed UDFs (prefix `lua_stat_`),
which call statcpp through a thin C binding. This path exists so that **new SQL
functions can be authored in Lua**, by composing statcpp primitives and encoding
policy, with **no C++ recompilation** — useful for developers who do not work in
C/C++ or for fast iteration.

```lua
-- src/lua/stats.lua — a NEW metric, composed in Lua (not a single statcpp call):
function lua_robust_cv(data)
    local scale = statcpp.mad(data)
    local loc   = statcpp.hodges_lehmann(data)
    if scale == nil or loc == nil or loc == 0 then return nil end
    return (1.4826 * scale) / loc          -- mad_scaled / Hodges-Lehmann
end

-- ...and a POLICY (business rule) that branches on the data:
function lua_smart_impute(data)
    local rate = statcpp.missing_rate(data)
    if rate == nil or rate > 0.5 then return nil end  -- refuse if too sparse
    if rate == 0 then return data end                 -- nothing to do
    return statcpp.impute_mean(data)                  -- else mean-impute
end
```

Each global `lua_<name>` in `stats.lua` is registered as the SQL function
`lua_stat_<name>`. Editing the script and re-running takes effect immediately
(the path is resolved at run time via `LUA_STATS_SCRIPT_PATH`), **as long as the
statcpp primitives used are already exposed by the binding**. Exposing a *new*
statcpp primitive to Lua is the one step that still needs a one-time C++ edit in
[lua_statcpp_bindings.hpp](src/include/lua_statcpp_bindings.hpp).

The bundled `stats.lua` ships three Lua-only functions with no C++ counterpart,
as worked examples of the pattern:

| Lua-backed UDF | Returns | Pattern shown |
| --- | --- | --- |
| `lua_stat_robust_cv` | scalar | Composition: `mad_scaled / Hodges-Lehmann` |
| `lua_stat_smart_impute` | list | Policy: refuse imputation if too sparse |
| `lua_stat_summary` | string | Free-form report: standard reps + custom fields |

The demo in [main.cpp](src/main.cpp) also asserts that the C++ and Lua paths
produce identical numbers (a `MATCH` column), so the Lua layer can be trusted as
a faithful front-end to statcpp.

## Architecture

```text
SQL query
   │  SELECT stat_median(list(v))  /  SELECT lua_stat_robust_cv(list(v))
   ▼
DuckDB UDF
   │  C++ path:  statcpp_udf.hpp  — table-driven {SQL name, lambda} registration
   │  Lua path:  lua_udf.hpp      — marshals LIST<DOUBLE> <-> Lua table, lua_pcall
   ▼
(Lua path only) src/lua/stats.lua          <-- authored by the extension developer
   │            function lua_<name>(data) ... statcpp.<fn>(data) ...
   ▼
(Lua path only) lua_statcpp_bindings.hpp   C binding: require("statcpp")
   ▼
statcpp (C++17, header-only)               the actual statistical computation
```

| File | Role |
| --- | --- |
| [statcpp_udf.hpp](src/include/statcpp_udf.hpp) | Direct C++ path. Table-driven registration of all `stat_*` functions. |
| [lua_udf.hpp](src/include/lua_udf.hpp) | Bridges DuckDB and Lua; registers each Lua function as a vectorized UDF. |
| [lua_statcpp_bindings.hpp](src/include/lua_statcpp_bindings.hpp) | One-time C binding exposing statcpp to Lua as `require("statcpp")`. |
| [src/lua/stats.lua](src/lua/stats.lua) | The file an extension developer edits — pure Lua. |
| [main.cpp](src/main.cpp) | Demo driver: registers both paths and walks each category. |

### Data / missing-value conventions

- DuckDB `LIST<DOUBLE>` ↔ a 1-indexed Lua table (Lua path) / `std::vector<double>`.
- SQL `NULL` ↔ statcpp `NaN` ↔ Lua `nil` (missing value), converted at each boundary.
- A `NaN` element in a transform's output becomes SQL `NULL`.
- Errors raised by statcpp or Lua are caught at the boundary and surfaced as SQL `NULL`.

## Build & run

Requirements: CMake 3.20+, a C++17 compiler (AppleClang), Ninja (used to build
DuckDB). DuckDB, statcpp and Lua (5.5.0) are all fetched from source on the first
configure — no system-installed Lua is required.

```bash
# Configure (first run downloads & builds DuckDB; this can take a while)
cmake -S . -B build

# Build
cmake --build build

# Run the demo
./build/a.out
```

If the dynamically linked DuckDB library is not found at run time, prefix the
command with its path:

```bash
DYLD_LIBRARY_PATH=download/DuckDB/DuckDB_C_CPP-install/lib ./build/a.out
```

## Project layout

```text
duckdb-lua-statcpp/
├── CMakeLists.txt
├── cmake/
│   ├── DuckDB_C_CPP.cmake     # Downloads, builds and links DuckDB (v1.4.4)
│   ├── statcpp.cmake          # Downloads statcpp headers (v0.2.0)
│   └── lua.cmake              # Downloads & builds Lua from source (v5.5.0)
├── src/
│   ├── main.cpp               # Demo: registers both paths, walks each category
│   ├── lua/
│   │   └── stats.lua          # Lua extension functions (edited by the developer)
│   └── include/
│       ├── statcpp_udf.hpp           # Direct C++ path (all stat_* functions)
│       ├── lua_statcpp_bindings.hpp  # C binding: statcpp -> Lua module
│       └── lua_udf.hpp               # Registers Lua functions as DuckDB UDFs
└── download/                  # Cached DuckDB / statcpp / Lua sources (git-ignored)
```

## Notes

- **Sorting.** Single-sample scalar functions sort the cleaned sample before the
  call, because several statcpp routines (`median`, `iqr`, `percentile`,
  `trimmed_mean`) require a sorted range. Sorting is harmless for order-independent
  statistics. Two-sample functions keep positional pairing; transforms keep order.
- **`stat_mad` is unscaled.** It returns the raw Median Absolute Deviation; the
  normal-consistent variant is `stat_mad_scaled` (`× 1.4826`).
- **Lua threading.** A single `lua_State` is shared across the Lua-backed UDFs and
  is **not thread-safe**. For concurrent query execution use per-thread states or a
  mutex. The C++ path has no such restriction.
- **Per-row overhead.** The Lua path performs a `lua_pcall` plus table marshalling
  per group/row, so it is heavier than the C++ path; prefer `stat_*` for hot paths
  and `lua_stat_*` for functions you want to author/iterate in Lua.

## License

This project is released under the **MIT License**.
Copyright (c) 2026 mitsuruk. See [LICENSE.md](LICENSE.md) for the full
terms.

Third-party dependencies keep their own licenses:

- [DuckDB](https://github.com/duckdb/duckdb) — MIT License
- [statcpp](https://github.com/mitsuruk/statcpp) — MIT License
- [Lua](https://www.lua.org/) — MIT License

DuckDB, statcpp and Lua are all fetched at build time (cached under
`download/`, git-ignored) and are **not redistributed** as part of this
repository.
