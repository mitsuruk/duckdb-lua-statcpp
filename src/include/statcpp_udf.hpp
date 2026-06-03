/**
 * @file statcpp_udf.hpp
 * @brief Register statcpp functions as DuckDB SQL UDFs (User-Defined Functions)
 *
 * Design
 * ------
 * Holistic statistics (median, MAD, percentile, correlation, ...) need the whole
 * sample at once, which does not fit DuckDB's fixed-size POD aggregate-state model.
 * So instead of CreateAggregateFunction we use the pattern:
 *
 *     aggregate a column into a LIST with list(), then pass the LIST to a
 *     vectorized scalar UDF that materializes it into std::vector<double> and
 *     forwards it to statcpp's iterator / vector based API.
 *
 * Registration is table-driven: each exposed function is a single
 * {SQL name, lambda} entry, so adding a statcpp function is a one-line change.
 *
 * Conventions at the UDF boundary
 * -------------------------------
 *  - DuckDB NULL  <->  statcpp missing value (NaN). A NULL LIST element becomes
 *    NaN on the way in; a NaN result element becomes SQL NULL on the way out.
 *  - For single-sample / two-sample statistics, missing values are dropped before
 *    the statistic is computed (pairwise for two-sample inputs).
 *  - statcpp throws on invalid input (empty range, etc.); every UDF catches at the
 *    boundary and returns SQL NULL, so no exception leaks into the DuckDB engine.
 *
 * For the meaning, assumptions and algorithms of each statistic, see the statcpp
 * documentation: https://github.com/<statcpp> (this layer only exposes them to SQL).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "duckdb.hpp"

#include "statcpp/statcpp.hpp"

namespace statcpp_duckdb {

/// Shorthand for the sample vector type passed to statcpp.
using Vec = std::vector<double>;

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Build the LIST<DOUBLE> logical type.
inline duckdb::LogicalType ListOfDouble() {
    return duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE);
}

/**
 * @brief Convert a DuckDB LIST value into std::vector<double>.
 *
 * A NULL LIST element is converted to NaN (DuckDB NULL -> statcpp missing value).
 * Returns an empty vector if the value itself is NULL.
 */
inline Vec ToVector(const duckdb::Value& list_val) {
    Vec out;
    if (list_val.IsNull()) {
        return out;
    }
    const auto& children = duckdb::ListValue::GetChildren(list_val);
    out.reserve(children.size());
    for (const auto& child : children) {
        if (child.IsNull()) {
            out.push_back(std::numeric_limits<double>::quiet_NaN());
        } else {
            out.push_back(child.GetValue<double>());
        }
    }
    return out;
}

/// Return a copy of `v` with all missing values (NaN) removed.
inline Vec DropNa(const Vec& v) {
    Vec clean;
    clean.reserve(v.size());
    for (const double x : v) {
        if (!statcpp::is_na(x)) {
            clean.push_back(x);
        }
    }
    return clean;
}

/// Build a DuckDB LIST<DOUBLE> value from a vector; NaN elements become SQL NULL.
inline duckdb::Value ToListValue(const Vec& v) {
    std::vector<duckdb::Value> children;
    children.reserve(v.size());
    for (const double d : v) {
        if (std::isnan(d)) {
            children.push_back(duckdb::Value(duckdb::LogicalType::DOUBLE));
        } else {
            children.push_back(duckdb::Value::DOUBLE(d));
        }
    }
    return duckdb::Value::LIST(duckdb::LogicalType::DOUBLE, std::move(children));
}

// ---------------------------------------------------------------------------
// UDF signatures (operate on plain vectors; the boundary glue lives below)
// ---------------------------------------------------------------------------

using ScalarFn = std::function<double(const Vec&)>;             ///< LIST<DOUBLE> -> DOUBLE
using ParamScalarFn = std::function<double(const Vec&, double)>;  ///< (LIST<DOUBLE>, DOUBLE) -> DOUBLE
using TwoListFn = std::function<double(const Vec&, const Vec&)>;  ///< (LIST<DOUBLE>, LIST<DOUBLE>) -> DOUBLE
using ListFn = std::function<Vec(const Vec&)>;                  ///< LIST<DOUBLE> -> LIST<DOUBLE>

// ---------------------------------------------------------------------------
// Registration helpers
// ---------------------------------------------------------------------------

/**
 * @brief Register a UDF LIST<DOUBLE> -> DOUBLE.
 * @param drop_na If true, missing values are removed and the sample is sorted before
 *        `fn` is called. Several statcpp statistics (median, iqr, ...) require a sorted
 *        range; sorting is harmless for order-independent ones (mean, variance, ...).
 *        Pass false to receive the raw, original-order list (e.g. for missing_rate).
 */
inline void RegisterListToScalar(duckdb::Connection& con, const std::string& name, ScalarFn fn,
                                 bool drop_na = true) {
    duckdb::scalar_function_t udf =
        [fn, drop_na](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
                      duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            Vec values = ToVector(args.data[0].GetValue(i));
            if (drop_na) {
                values = DropNa(values);
                std::sort(values.begin(), values.end());
            }
            try {
                result.SetValue(i, duckdb::Value::DOUBLE(fn(values)));
            } catch (const std::exception&) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
            }
        }
    };
    con.CreateVectorizedFunction(name, {ListOfDouble()}, duckdb::LogicalType::DOUBLE, udf);
}

/**
 * @brief Register a UDF (LIST<DOUBLE>, DOUBLE) -> DOUBLE.
 *
 * Missing values are removed and the sample is sorted before `fn` is called (the
 * parameterized statistics here — percentile, trimmed_mean — index a sorted range).
 */
inline void RegisterListParamToScalar(duckdb::Connection& con, const std::string& name,
                                      ParamScalarFn fn) {
    duckdb::scalar_function_t udf =
        [fn](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
             duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            Vec values = DropNa(ToVector(args.data[0].GetValue(i)));
            std::sort(values.begin(), values.end());
            const duckdb::Value param = args.data[1].GetValue(i);
            if (param.IsNull()) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
                continue;
            }
            try {
                result.SetValue(i, duckdb::Value::DOUBLE(fn(values, param.GetValue<double>())));
            } catch (const std::exception&) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
            }
        }
    };
    con.CreateVectorizedFunction(name, {ListOfDouble(), duckdb::LogicalType::DOUBLE},
                                 duckdb::LogicalType::DOUBLE, udf);
}

/**
 * @brief Register a UDF (LIST<DOUBLE>, LIST<DOUBLE>) -> DOUBLE.
 *
 * Pairs where either element is missing are dropped before `fn` is called, so the
 * two cleaned vectors always have equal length.
 */
inline void RegisterTwoListToScalar(duckdb::Connection& con, const std::string& name,
                                    TwoListFn fn) {
    duckdb::scalar_function_t udf =
        [fn](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
             duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            const Vec x = ToVector(args.data[0].GetValue(i));
            const Vec y = ToVector(args.data[1].GetValue(i));
            const std::size_t n = std::min(x.size(), y.size());
            Vec cx;
            Vec cy;
            cx.reserve(n);
            cy.reserve(n);
            for (std::size_t k = 0; k < n; ++k) {
                if (!statcpp::is_na(x[k]) && !statcpp::is_na(y[k])) {
                    cx.push_back(x[k]);
                    cy.push_back(y[k]);
                }
            }
            try {
                result.SetValue(i, duckdb::Value::DOUBLE(fn(cx, cy)));
            } catch (const std::exception&) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
            }
        }
    };
    con.CreateVectorizedFunction(name, {ListOfDouble(), ListOfDouble()},
                                 duckdb::LogicalType::DOUBLE, udf);
}

/**
 * @brief Register a UDF LIST<DOUBLE> -> LIST<DOUBLE> (column -> column transform).
 *
 * The raw vector (missing values preserved as NaN) is passed to `fn`; this is what
 * imputation / transform functions need. NaN elements in the output become SQL NULL.
 */
inline void RegisterListToList(duckdb::Connection& con, const std::string& name, ListFn fn) {
    duckdb::scalar_function_t udf =
        [fn](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
             duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            const Vec values = ToVector(args.data[0].GetValue(i));
            try {
                result.SetValue(i, ToListValue(fn(values)));
            } catch (const std::exception&) {
                result.SetValue(i, duckdb::Value(ListOfDouble()));
            }
        }
    };
    con.CreateVectorizedFunction(name, {ListOfDouble()}, ListOfDouble(), udf);
}

// ---------------------------------------------------------------------------
// Function tables
// ---------------------------------------------------------------------------

/**
 * @brief Single-sample statistics: LIST<DOUBLE> -> DOUBLE (missing values dropped).
 *
 * Each lambda receives the cleaned (NaN-free) sample. Names mirror statcpp with a
 * "stat_" prefix.
 */
inline const std::vector<std::pair<std::string, ScalarFn>>& ScalarFunctionTable() {
    static const std::vector<std::pair<std::string, ScalarFn>> table = {
        // --- basic statistics ---
        {"stat_sum", [](const Vec& v) { return statcpp::sum(v.begin(), v.end()); }},
        {"stat_count", [](const Vec& v) { return static_cast<double>(statcpp::count(v.begin(), v.end())); }},
        {"stat_mean", [](const Vec& v) { return statcpp::mean(v.begin(), v.end()); }},
        {"stat_median", [](const Vec& v) { return statcpp::median(v.begin(), v.end()); }},
        {"stat_mode", [](const Vec& v) { return static_cast<double>(statcpp::mode(v.begin(), v.end())); }},
        {"stat_geometric_mean", [](const Vec& v) { return statcpp::geometric_mean(v.begin(), v.end()); }},
        {"stat_harmonic_mean", [](const Vec& v) { return statcpp::harmonic_mean(v.begin(), v.end()); }},
        // --- dispersion / spread ---
        {"stat_range", [](const Vec& v) { return statcpp::range(v.begin(), v.end()); }},
        {"stat_variance", [](const Vec& v) { return statcpp::variance(v.begin(), v.end()); }},
        {"stat_population_variance", [](const Vec& v) { return statcpp::population_variance(v.begin(), v.end()); }},
        {"stat_sample_variance", [](const Vec& v) { return statcpp::sample_variance(v.begin(), v.end()); }},
        {"stat_stddev", [](const Vec& v) { return statcpp::stddev(v.begin(), v.end()); }},
        {"stat_population_stddev", [](const Vec& v) { return statcpp::population_stddev(v.begin(), v.end()); }},
        {"stat_sample_stddev", [](const Vec& v) { return statcpp::sample_stddev(v.begin(), v.end()); }},
        {"stat_coefficient_of_variation",
         [](const Vec& v) { return statcpp::coefficient_of_variation(v.begin(), v.end()); }},
        {"stat_iqr", [](const Vec& v) { return statcpp::iqr(v.begin(), v.end()); }},
        // --- shape of distribution ---
        {"stat_skewness", [](const Vec& v) { return statcpp::skewness(v.begin(), v.end()); }},
        {"stat_sample_skewness", [](const Vec& v) { return statcpp::sample_skewness(v.begin(), v.end()); }},
        {"stat_population_skewness", [](const Vec& v) { return statcpp::population_skewness(v.begin(), v.end()); }},
        {"stat_kurtosis", [](const Vec& v) { return statcpp::kurtosis(v.begin(), v.end()); }},
        {"stat_sample_kurtosis", [](const Vec& v) { return statcpp::sample_kurtosis(v.begin(), v.end()); }},
        {"stat_population_kurtosis", [](const Vec& v) { return statcpp::population_kurtosis(v.begin(), v.end()); }},
        // --- order statistics ---
        {"stat_minimum", [](const Vec& v) { return static_cast<double>(statcpp::minimum(v.begin(), v.end())); }},
        {"stat_maximum", [](const Vec& v) { return static_cast<double>(statcpp::maximum(v.begin(), v.end())); }},
        // --- robust ---
        {"stat_mad", [](const Vec& v) { return statcpp::mad(v.begin(), v.end()); }},
        {"stat_mad_scaled", [](const Vec& v) { return statcpp::mad_scaled(v.begin(), v.end()); }},
        {"stat_hodges_lehmann", [](const Vec& v) { return statcpp::hodges_lehmann(v.begin(), v.end()); }},
        {"stat_biweight_midvariance",
         [](const Vec& v) { return statcpp::biweight_midvariance(v.begin(), v.end()); }},
    };
    return table;
}

/**
 * @brief Single-sample statistics with one scalar parameter: (LIST<DOUBLE>, DOUBLE) -> DOUBLE.
 *
 * The sample is already cleaned and sorted by RegisterListParamToScalar.
 */
inline const std::vector<std::pair<std::string, ParamScalarFn>>& ParamScalarFunctionTable() {
    static const std::vector<std::pair<std::string, ParamScalarFn>> table = {
        // p is a proportion in [0, 1] (e.g. 0.9 -> 90th percentile)
        {"stat_percentile",
         [](const Vec& v, double p) { return statcpp::percentile(v.begin(), v.end(), p); }},
        // proportion is the fraction trimmed per side, in [0, 0.5)
        {"stat_trimmed_mean",
         [](const Vec& v, double proportion) { return statcpp::trimmed_mean(v.begin(), v.end(), proportion); }},
    };
    return table;
}

/**
 * @brief Two-sample statistics: (LIST<DOUBLE>, LIST<DOUBLE>) -> DOUBLE.
 *
 * Inputs are aligned by position; pairs with a missing value are dropped before
 * the call (see RegisterTwoListToScalar).
 */
inline const std::vector<std::pair<std::string, TwoListFn>>& TwoListFunctionTable() {
    static const std::vector<std::pair<std::string, TwoListFn>> table = {
        {"stat_covariance",
         [](const Vec& x, const Vec& y) { return statcpp::covariance(x.begin(), x.end(), y.begin(), y.end()); }},
        {"stat_sample_covariance",
         [](const Vec& x, const Vec& y) {
             return statcpp::sample_covariance(x.begin(), x.end(), y.begin(), y.end());
         }},
        {"stat_population_covariance",
         [](const Vec& x, const Vec& y) {
             return statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end());
         }},
        {"stat_pearson_correlation",
         [](const Vec& x, const Vec& y) {
             return statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end());
         }},
        {"stat_spearman_correlation",
         [](const Vec& x, const Vec& y) {
             return statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end());
         }},
        {"stat_kendall_tau",
         [](const Vec& x, const Vec& y) { return statcpp::kendall_tau(x.begin(), x.end(), y.begin(), y.end()); }},
    };
    return table;
}

/**
 * @brief Column transforms: LIST<DOUBLE> -> LIST<DOUBLE> (missing values preserved as input).
 *
 * fillna_* treat NaN as the value to fill; the math transforms map element-wise.
 */
inline const std::vector<std::pair<std::string, ListFn>>& ListFunctionTable() {
    static const std::vector<std::pair<std::string, ListFn>> table = {
        {"stat_fillna_mean", [](const Vec& v) { return statcpp::fillna_mean(v); }},
        {"stat_fillna_median", [](const Vec& v) { return statcpp::fillna_median(v); }},
        {"stat_fillna_ffill", [](const Vec& v) { return statcpp::fillna_ffill(v); }},
        {"stat_fillna_bfill", [](const Vec& v) { return statcpp::fillna_bfill(v); }},
        {"stat_fillna_interpolate", [](const Vec& v) { return statcpp::fillna_interpolate(v); }},
        {"stat_log_transform", [](const Vec& v) { return statcpp::log_transform(v); }},
        {"stat_sqrt_transform", [](const Vec& v) { return statcpp::sqrt_transform(v); }},
        {"stat_rank_transform", [](const Vec& v) { return statcpp::rank_transform(v); }},
        {"stat_winsorize", [](const Vec& v) { return statcpp::winsorize(v.begin(), v.end()); }},
    };
    return table;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief Register all statcpp UDFs on the given connection.
 *
 * After this call the statistics are available as SQL functions, typically applied
 * to a LIST built with DuckDB's list() aggregate, e.g.
 *
 *     SELECT stat_median(list(v)) FROM t;
 *     SELECT stat_percentile(list(v), 0.9) FROM t;
 *     SELECT stat_pearson_correlation(list(x), list(y)) FROM t;
 *     SELECT unnest(stat_fillna_mean(list(v ORDER BY id))) FROM t;
 *
 * In addition to the table-driven functions above, two custom missing-data helpers
 * are kept for backward compatibility:
 *  - stat_missing_rate(LIST<DOUBLE>) -> DOUBLE        : fraction of missing values
 *  - stat_impute_mean(LIST<DOUBLE>)  -> LIST<DOUBLE>  : mean imputation
 */
inline void RegisterStatcppFunctions(duckdb::Connection& con) {
    for (const auto& [name, fn] : ScalarFunctionTable()) {
        RegisterListToScalar(con, name, fn);
    }
    for (const auto& [name, fn] : ParamScalarFunctionTable()) {
        RegisterListParamToScalar(con, name, fn);
    }
    for (const auto& [name, fn] : TwoListFunctionTable()) {
        RegisterTwoListToScalar(con, name, fn);
    }
    for (const auto& [name, fn] : ListFunctionTable()) {
        RegisterListToList(con, name, fn);
    }

    // --- custom missing-data helpers (no direct one-call statcpp counterpart) ---

    // stat_missing_rate: fraction of missing values, via analyze_missing_patterns.
    // Needs to see the missing values, so it operates on the raw list (drop_na = false).
    RegisterListToScalar(
        con, "stat_missing_rate",
        [](const Vec& v) -> double {
            std::vector<Vec> matrix;
            matrix.reserve(v.size());
            for (const double x : v) {
                matrix.push_back({x});
            }
            const auto info = statcpp::analyze_missing_patterns(matrix);
            return info.missing_rates.at(0);
        },
        /*drop_na=*/false);

    // stat_impute_mean: fill each missing value with the observed mean.
    RegisterListToList(con, "stat_impute_mean", [](const Vec& v) -> Vec {
        const Vec observed = DropNa(v);
        const double fill = observed.empty() ? std::numeric_limits<double>::quiet_NaN()
                                             : statcpp::mean(observed.begin(), observed.end());
        Vec out;
        out.reserve(v.size());
        for (const double x : v) {
            out.push_back(statcpp::is_na(x) ? fill : x);
        }
        return out;
    });
}

}  // namespace statcpp_duckdb
