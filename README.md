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
DuckDB connection **200+ statistical SQL functions** — descriptive statistics,
order statistics, robust estimators, correlation/covariance, weighted statistics,
error/distance metrics, window/rolling statistics, time-series transforms,
probability distributions (pdf/cdf/quantile/rand), special functions, effect
sizes and power analysis — all computed by statcpp. The same data path is also
reachable through Lua, so new SQL functions can be composed in a script and take
effect on the next run with no rebuild.

Function names mirror those of the sibling project
[sqlite3-stats](https://github.com/mitsuruk/sqlite3StatisticalLibrary); this PoC
covers the same set **except JSON-returning functions** (hypothesis-test result
objects, frequency tables, survival curves, CI/regression objects), which are out
of scope here.

Released under the [MIT License](LICENSE.md). DuckDB, statcpp and Lua are
third-party dependencies under their own licenses — see [License](#license).

## What you get

Two header-only layers you register on a `duckdb::Connection`:

```cpp
#include "statcpp_udf.hpp"     // LIST-based UDFs: aggregates, two-column, window
#include "statcpp_scalar.hpp"  // scalar UDFs: distributions, special functions, effect size
#include "lua_udf.hpp"         // optional Lua path: author new functions in Lua

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
       stat_stdev(list(v))           AS stdev,
       stat_hodges_lehmann(list(v))  AS robust_location,
       stat_mad(list(v))             AS robust_scale,
       stat_percentile(list(v), 0.9) AS p90
FROM measurements
GROUP BY grp;

-- Correlation between two columns
SELECT stat_pearson_r(list(x), list(y)) AS r FROM paired;

-- Rolling mean (window) over an ordered list, expanded back to rows
SELECT unnest(stat_rolling_mean(list(v ORDER BY id), 3)) AS ma3 FROM measurements;

-- Scalar: 97.5th percentile of the standard normal (≈ 1.96)
SELECT stat_normal_quantile(0.975, 0, 1) AS z_975;
```

> **Statistics reference.** This repository only *exposes* statcpp to SQL. For
> the definition, assumptions, parameters and algorithm of each statistic, see
> the [statcpp documentation](https://github.com/mitsuruk/statcpp). Names below
> mirror statcpp with a `stat_` prefix.

## Function catalog

**200+ functions** in two families: **LIST-based** (a column is first aggregated
with `list()`) and **scalar** (plain `DOUBLE` arguments). Missing values
(SQL `NULL` / `NaN`) are dropped before single- and two-sample statistics
(pairwise for two-sample); on invalid input the function returns SQL `NULL`.
Names mirror [sqlite3-stats](https://github.com/mitsuruk/sqlite3StatisticalLibrary).

### Basic aggregates — `(LIST<DOUBLE>) → DOUBLE`

| Category | Functions |
| --- | --- |
| Central tendency | `stat_mean` `stat_median` `stat_mode` `stat_geometric_mean` `stat_harmonic_mean` |
| Dispersion | `stat_range` `stat_var` `stat_population_variance` `stat_sample_variance` `stat_stdev` `stat_population_stddev` `stat_sample_stddev` `stat_cv` `stat_iqr` `stat_mad_mean` `stat_geometric_stddev` |
| Shape | `stat_skewness` `stat_population_skewness` `stat_kurtosis` `stat_population_kurtosis` |
| Estimation / robust | `stat_se` `stat_mad` `stat_mad_scaled` `stat_hodges_lehmann` |
| Extras (no sqlite3 equiv.) | `stat_sum` `stat_count` `stat_minimum` `stat_maximum` |

### Parameterized aggregates — `(LIST<DOUBLE>, DOUBLE) → DOUBLE`

`stat_trimmed_mean` `stat_percentile` `stat_moe_mean` `stat_cohens_d`
`stat_hedges_g` `stat_acf_lag` `stat_biweight_midvar`

### Two-column aggregates — `(LIST<DOUBLE>, LIST<DOUBLE>[, DOUBLE]) → DOUBLE`

| Category | Functions |
| --- | --- |
| Correlation / covariance | `stat_covariance` `stat_population_covariance` `stat_pearson_r` `stat_spearman_r` `stat_kendall_tau` `stat_weighted_covariance` |
| Weighted statistics | `stat_weighted_mean` `stat_weighted_harmonic_mean` `stat_weighted_variance` `stat_weighted_stddev` `stat_weighted_median` `stat_weighted_percentile` |
| Regression (scalar) | `stat_r_squared` `stat_adjusted_r_squared` |
| Error metrics | `stat_mae` `stat_mse` `stat_rmse` `stat_mape` |
| Distance metrics | `stat_euclidean_dist` `stat_manhattan_dist` `stat_cosine_sim` `stat_cosine_dist` `stat_minkowski_dist` `stat_chebyshev_dist` |
| Two-sample effect sizes | `stat_cohens_d2` `stat_hedges_g2` `stat_glass_delta` |

### Window / transforms — `(LIST<DOUBLE>[, DOUBLE]) → LIST<DOUBLE>`

| Category | Functions |
| --- | --- |
| Rolling / moving | `stat_rolling_mean` `stat_rolling_std` `stat_rolling_min` `stat_rolling_max` `stat_rolling_sum` `stat_moving_avg` `stat_ema` |
| Rank / fill | `stat_rank` `stat_fillna_mean` `stat_fillna_median` `stat_fillna_ffill` `stat_fillna_bfill` `stat_fillna_interp` |
| Encoding / binning | `stat_label_encode` `stat_bin_width` `stat_bin_freq` |
| Time series | `stat_lag` `stat_diff` `stat_seasonal_diff` |
| Outliers / robust | `stat_outliers_iqr` `stat_outliers_zscore` `stat_outliers_mzscore` `stat_winsorize` |
| Extras | `stat_impute_mean` `stat_log_transform` `stat_sqrt_transform` `stat_missing_rate` *(→ scalar)* |

Wrap a window/transform result in `unnest()` to expand it back into rows; keep
input order with `list(col ORDER BY key)`.

### Scalar — distributions — `(DOUBLE…) → DOUBLE`

Each distribution provides `pdf`/`pmf`, `cdf`, `quantile` and `rand`:

`stat_normal_*` `stat_lognormal_*` `stat_uniform_*` `stat_exponential_*`
`stat_gamma_*` `stat_beta_*` `stat_weibull_*` `stat_chisq_*` `stat_t_*`
`stat_f_*` `stat_binomial_*` `stat_poisson_*` `stat_geometric_*`
`stat_nbinom_*` `stat_hypergeom_*` `stat_bernoulli_*` `stat_duniform_*`

### Scalar — special / tests / effect size — `(DOUBLE…) → DOUBLE | VARCHAR`

| Category | Functions |
| --- | --- |
| Special functions | `stat_lgamma` `stat_tgamma` `stat_beta_func` `stat_lbeta` `stat_erf` `stat_erfc` `stat_betainc` `stat_betaincinv` `stat_norm_cdf` `stat_norm_quantile` `stat_gammainc_lower` `stat_gammainc_upper` `stat_gammainc_lower_inv` |
| Combinatorics | `stat_binomial_coef` `stat_log_binomial_coef` `stat_log_factorial` |
| Tests / corrections | `stat_bonferroni` `stat_bh_correction` `stat_holm_correction` `stat_nnt` `stat_aic` `stat_aicc` `stat_bic` `stat_boxcox` `stat_logarithmic_mean` |
| Effect size | `stat_hedges_j` `stat_t_to_r` `stat_d_to_r` `stat_r_to_d` `stat_eta_squared_ef` `stat_partial_eta_sq` `stat_omega_squared_ef` `stat_cohens_h` |
| Interpretation (`→ VARCHAR`) | `stat_interpret_d` `stat_interpret_r` `stat_interpret_eta2` |
| Power / sample size | `stat_power_t1` `stat_n_t1` `stat_power_t2` `stat_n_t2` `stat_power_prop` `stat_n_prop` `stat_moe_prop` `stat_moe_prop_worst` `stat_n_moe_prop` `stat_n_moe_mean` |

### Scope

This PoC exposes every sqlite3-stats function that fits a `DOUBLE` / `LIST` /
`VARCHAR-label` signature. **JSON-returning functions are out of scope** —
hypothesis-test result objects (`stat_t_test`, `stat_anova1`, …), frequency
tables, contingency tables, survival curves (Kaplan–Meier / Nelson–Aalen),
confidence-interval and regression objects, and bootstrap distributions. Call
statcpp directly in C++ for those.

> **Scalar arities are fixed.** Every scalar function takes all of its arguments
> explicitly (e.g. `stat_normal_pdf(x, mu, sigma)` — no defaulted `mu`/`sigma`).
> Integer-valued arguments are passed as `DOUBLE` and cast internally.
>
> **Random functions** (`*_rand`) are registered as ordinary (deterministic) UDFs
> because the DuckDB C++ UDF API exposes no volatility flag; calls with only
> constant arguments may be constant-folded to a single value per query.

### Differences from sqlite3-stats

Function names and numerical results mirror
[sqlite3-stats](https://github.com/mitsuruk/sqlite3StatisticalLibrary). The
differences are in *which* functions exist and *how* they are invoked, not in the
math:

| Aspect | sqlite3-stats | this PoC (duckdb) |
| --- | --- | --- |
| JSON-returning functions | included (test/CI/regression/survival/frequency objects as JSON) | **omitted** — out of scope (see [Scope](#scope)) |
| Window / rolling | true SQL window functions: `stat_rolling_mean(v, 3) OVER (ORDER BY t)` | `(LIST) → LIST` transforms: `unnest(stat_rolling_mean(list(v ORDER BY t), 3))` |
| Aggregates | native aggregates: `stat_mean(v)` | applied to a `list()`: `stat_mean(list(v))` |
| Scalar argument defaults | optional args with defaults (e.g. `stat_normal_pdf(x)` ⇒ `mu=0, sigma=1`) | **fixed arity**, all args required |
| `*_rand` volatility | registered non-deterministic | registered deterministic (DuckDB C++ UDF API limitation) |
| Renamed for clarity? | — | names are **identical** to sqlite3-stats (`stat_stdev`, `stat_var`, `stat_pearson_r`, `stat_fillna_interp`, …) |

The following **statistical conventions are inherited unchanged** from
sqlite3-stats / statcpp, and are called out here only because they can surprise:

- **`stat_stdev` / `stat_var` use `ddof = 0` (population).** They are equivalent to
  `stat_population_stddev` / `stat_population_variance`. For the sample estimators
  (`ddof = 1`) use `stat_sample_stddev` / `stat_sample_variance`. Example:
  for `{9, 10, 10.5, 11, 1000}`, `stat_stdev = 395.95` (population), while
  `stat_sample_stddev = 442.69`.
- **Rolling/window functions are leading-aligned.** `stat_rolling_mean(list(v), w)`
  places each window result at the *first* position of its window, so the **last
  `w − 1` rows are `NULL`** (not the first `w − 1`). The values themselves are the
  means/sums of consecutive `w`-element runs.

> **This is a PoC, so these choices are not set in stone.** Any of the differences
> above can be revisited on request — e.g. trailing-aligned rolling, `ddof = 1`
> defaults for `stat_stdev`/`stat_var`, defaulted scalar arguments, true window
> functions, or exposing the JSON-returning functions. Open an issue or ask.

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
| [statcpp_compute.hpp](src/include/statcpp_compute.hpp) | DB-agnostic pure compute functions (`compute::*`) ported from statcpp. |
| [statcpp_udf.hpp](src/include/statcpp_udf.hpp) | LIST-based UDFs: registration helpers + aggregate / two-column / window tables. |
| [statcpp_scalar.hpp](src/include/statcpp_scalar.hpp) | Scalar `(DOUBLE…)` UDFs: distributions, special functions, effect size; plus the `RegisterStatcppFunctions` umbrella. |
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
│       ├── statcpp_compute.hpp       # Pure compute functions (DB-agnostic)
│       ├── statcpp_udf.hpp           # LIST-based UDFs (aggregate/two-col/window)
│       ├── statcpp_scalar.hpp        # Scalar UDFs (distributions/effect size) + umbrella
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
