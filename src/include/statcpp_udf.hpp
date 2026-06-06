/**
 * @file statcpp_udf.hpp
 * @brief Register statcpp functions as DuckDB SQL UDFs (aggregate / two-column / window / extras)
 *
 * Design
 * ------
 * Holistic statistics (median, MAD, percentile, correlation, ...) need the whole
 * sample at once, which does not fit DuckDB's fixed-size POD aggregate-state model.
 * So instead of CreateAggregateFunction we use the pattern:
 *
 *     aggregate a column into a LIST with list(), then pass the LIST to a
 *     vectorized scalar UDF that materializes it into std::vector<double> and
 *     forwards it to statcpp's vector / iterator API.
 *
 * Registration is table-driven: each exposed function is a single
 * {SQL name, compute function} entry. The compute bodies live in the pure
 * functions of statcpp_compute.hpp (compute::*).
 *
 * Conventions at the UDF boundary
 * -------------------------------
 *  - DuckDB NULL <-> statcpp missing value (NaN). A NULL LIST element becomes NaN
 *    on the way in; a NaN result element becomes SQL NULL on the way out.
 *  - For single-sample / two-sample statistics, missing values are dropped before
 *    the statistic is computed (pairwise for two-sample inputs).
 *  - statcpp may throw on invalid input; every UDF catches at the boundary and
 *    returns SQL NULL, so no exception leaks into the DuckDB engine.
 *
 * Scalar functions (distributions / test helpers) are registered separately in
 * statcpp_scalar.hpp.
 *
 * For the meaning, assumptions and algorithm of each statistic, see the statcpp
 * documentation: https://github.com/mitsuruk/statcpp (this layer only exposes them to SQL).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "duckdb.hpp"

#include "statcpp/statcpp.hpp"
#include "statcpp_compute.hpp"

namespace statcpp_duckdb {

/// Sample vector type passed to statcpp.
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

using ScalarFn = std::function<double(const Vec&)>;                ///< LIST<DOUBLE> -> DOUBLE
using ParamScalarFn = std::function<double(const Vec&, double)>;   ///< (LIST<DOUBLE>, DOUBLE) -> DOUBLE
using TwoListFn = std::function<double(const Vec&, const Vec&)>;   ///< (LIST, LIST) -> DOUBLE
using TwoListParamFn =
    std::function<double(const Vec&, const Vec&, double)>;          ///< (LIST, LIST, DOUBLE) -> DOUBLE
using ListFn = std::function<Vec(const Vec&)>;                     ///< LIST -> LIST
using WindowFn =
    std::function<Vec(const Vec&, const std::vector<bool>&, int)>;  ///< (LIST[, DOUBLE]) -> LIST

// ---------------------------------------------------------------------------
// Registration helpers
// ---------------------------------------------------------------------------

/**
 * @brief Register a UDF LIST<DOUBLE> -> DOUBLE.
 * @param drop_na If true, missing values are removed and the sample is sorted before
 *        `fn` is called. Basic statistics are order-independent, so sorting is harmless
 *        and also serves as the prerequisite for median/iqr/etc. Pass false when the raw
 *        order is needed (e.g. missing rate).
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
 * @param sort If true, the cleaned sample is sorted before `fn` is called
 *        (percentile / trimmed_mean / etc.). Order-dependent statistics
 *        (autocorrelation, ...) must pass false.
 */
inline void RegisterListParamToScalar(duckdb::Connection& con, const std::string& name,
                                      ParamScalarFn fn, bool sort = true) {
    duckdb::scalar_function_t udf =
        [fn, sort](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
                   duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            Vec values = DropNa(ToVector(args.data[0].GetValue(i)));
            if (sort) {
                std::sort(values.begin(), values.end());
            }
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

/// Pair two LISTs by position and keep only the pairs where neither element is missing.
inline void PairwiseClean(const duckdb::Value& xv, const duckdb::Value& yv, Vec& cx, Vec& cy) {
    const Vec x = ToVector(xv);
    const Vec y = ToVector(yv);
    const std::size_t n = std::min(x.size(), y.size());
    cx.reserve(n);
    cy.reserve(n);
    for (std::size_t k = 0; k < n; ++k) {
        if (!statcpp::is_na(x[k]) && !statcpp::is_na(y[k])) {
            cx.push_back(x[k]);
            cy.push_back(y[k]);
        }
    }
}

/// Register a UDF (LIST<DOUBLE>, LIST<DOUBLE>) -> DOUBLE (drops missing pairs).
inline void RegisterTwoListToScalar(duckdb::Connection& con, const std::string& name, TwoListFn fn) {
    duckdb::scalar_function_t udf =
        [fn](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
             duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            Vec cx;
            Vec cy;
            PairwiseClean(args.data[0].GetValue(i), args.data[1].GetValue(i), cx, cy);
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

/// Register a UDF (LIST<DOUBLE>, LIST<DOUBLE>, DOUBLE) -> DOUBLE (drops missing pairs).
inline void RegisterTwoListParamToScalar(duckdb::Connection& con, const std::string& name,
                                         TwoListParamFn fn) {
    duckdb::scalar_function_t udf =
        [fn](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
             duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            Vec cx;
            Vec cy;
            PairwiseClean(args.data[0].GetValue(i), args.data[1].GetValue(i), cx, cy);
            const duckdb::Value param = args.data[2].GetValue(i);
            if (param.IsNull()) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
                continue;
            }
            try {
                result.SetValue(i, duckdb::Value::DOUBLE(fn(cx, cy, param.GetValue<double>())));
            } catch (const std::exception&) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
            }
        }
    };
    con.CreateVectorizedFunction(
        name, {ListOfDouble(), ListOfDouble(), duckdb::LogicalType::DOUBLE},
        duckdb::LogicalType::DOUBLE, udf);
}

/**
 * @brief Register a UDF LIST<DOUBLE> -> LIST<DOUBLE> (column -> column transform).
 *
 * The raw vector (missing values preserved as NaN) is passed to `fn`. NaN elements in
 * the output become SQL NULL.
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

/**
 * @brief Register a window-style UDF.
 *
 * sqlite3-stats' true window functions are modeled in DuckDB as a LIST -> LIST
 * transform: the column is aggregated with list() and the same-length result LIST is
 * expanded back into rows with unnest(). A NULL element in the input LIST is treated as
 * a missing flag (NaN) for `fn`.
 * @param has_param If true, the second DOUBLE argument (window size / lag / ...) is
 *        passed to `fn` as an int.
 */
inline void RegisterWindow(duckdb::Connection& con, const std::string& name, WindowFn fn,
                           bool has_param) {
    duckdb::scalar_function_t udf =
        [fn, has_param](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
                        duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            const Vec values = ToVector(args.data[0].GetValue(i));
            std::vector<bool> nulls(values.size());
            for (std::size_t k = 0; k < values.size(); ++k) {
                nulls[k] = std::isnan(values[k]);
            }
            int param = 0;
            if (has_param) {
                const duckdb::Value pv = args.data[1].GetValue(i);
                if (!pv.IsNull()) {
                    param = static_cast<int>(pv.GetValue<double>());
                }
            }
            try {
                result.SetValue(i, ToListValue(fn(values, nulls, param)));
            } catch (const std::exception&) {
                result.SetValue(i, duckdb::Value(ListOfDouble()));
            }
        }
    };
    duckdb::vector<duckdb::LogicalType> arg_types = {ListOfDouble()};
    if (has_param) {
        arg_types.push_back(duckdb::LogicalType::DOUBLE);
    }
    con.CreateVectorizedFunction(name, arg_types, ListOfDouble(), udf);
}

// ---------------------------------------------------------------------------
// Function tables
// ---------------------------------------------------------------------------

namespace cc = statcpp_duckdb::compute;

/// Basic aggregates: LIST<DOUBLE> -> DOUBLE (24 functions; called NaN-free and sorted).
inline const std::vector<std::pair<std::string, ScalarFn>>& ScalarFunctionTable() {
    static const std::vector<std::pair<std::string, ScalarFn>> table = {
        // Central tendency
        {"stat_mean", cc::Mean},
        {"stat_median", cc::Median},
        {"stat_mode", cc::Mode},
        {"stat_geometric_mean", cc::GeometricMean},
        {"stat_harmonic_mean", cc::HarmonicMean},
        // Dispersion / spread
        {"stat_range", cc::Range},
        {"stat_var", cc::Var},
        {"stat_population_variance", cc::PopulationVariance},
        {"stat_sample_variance", cc::SampleVariance},
        {"stat_stdev", cc::Stdev},
        {"stat_population_stddev", cc::PopulationStddev},
        {"stat_sample_stddev", cc::SampleStddev},
        {"stat_cv", cc::Cv},
        {"stat_iqr", cc::Iqr},
        {"stat_mad_mean", cc::MadMean},
        {"stat_geometric_stddev", cc::GeometricStddev},
        // Shape of distribution
        {"stat_population_skewness", cc::PopulationSkewness},
        {"stat_skewness", cc::Skewness},
        {"stat_population_kurtosis", cc::PopulationKurtosis},
        {"stat_kurtosis", cc::Kurtosis},
        // Estimation
        {"stat_se", cc::Se},
        // Robust statistics
        {"stat_mad", cc::Mad},
        {"stat_mad_scaled", cc::MadScaled},
        {"stat_hodges_lehmann", cc::HodgesLehmann},
    };
    return table;
}

/// Parameterized aggregates: (LIST<DOUBLE>, DOUBLE) -> DOUBLE (7 functions).
/// pair: {name, {compute function, needs_sort}}
inline const std::vector<std::pair<std::string, std::pair<ParamScalarFn, bool>>>&
ParamScalarFunctionTable() {
    static const std::vector<std::pair<std::string, std::pair<ParamScalarFn, bool>>> table = {
        {"stat_trimmed_mean", {cc::TrimmedMean, true}},
        {"stat_percentile", {cc::Percentile, true}},
        {"stat_moe_mean", {cc::MoeMean, true}},
        {"stat_cohens_d", {cc::CohensD, true}},
        {"stat_hedges_g", {cc::HedgesG, true}},
        {"stat_acf_lag", {cc::AcfLag, false}},  // time series: keep order
        {"stat_biweight_midvar", {cc::BiweightMidvar, true}},
    };
    return table;
}

/// Two-column aggregates: (LIST, LIST) -> DOUBLE (22 functions + 3 effect sizes).
inline const std::vector<std::pair<std::string, TwoListFn>>& TwoListFunctionTable() {
    static const std::vector<std::pair<std::string, TwoListFn>> table = {
        // Correlation / covariance
        {"stat_population_covariance", cc::PopulationCovariance},
        {"stat_covariance", cc::Covariance},
        {"stat_pearson_r", cc::PearsonR},
        {"stat_spearman_r", cc::SpearmanR},
        {"stat_kendall_tau", cc::KendallTau},
        {"stat_weighted_covariance", cc::WeightedCovariance},
        // Weighted statistics
        {"stat_weighted_mean", cc::WeightedMean},
        {"stat_weighted_harmonic_mean", cc::WeightedHarmonicMean},
        {"stat_weighted_variance", cc::WeightedVariance},
        {"stat_weighted_stddev", cc::WeightedStddev},
        {"stat_weighted_median", cc::WeightedMedian},
        // Regression (double)
        {"stat_r_squared", cc::RSquared},
        {"stat_adjusted_r_squared", cc::AdjustedRSquared},
        // Error metrics
        {"stat_mae", cc::Mae},
        {"stat_mse", cc::Mse},
        {"stat_rmse", cc::Rmse},
        {"stat_mape", cc::Mape},
        // Distance metrics
        {"stat_euclidean_dist", cc::EuclideanDist},
        {"stat_manhattan_dist", cc::ManhattanDist},
        {"stat_cosine_sim", cc::CosineSim},
        {"stat_cosine_dist", cc::CosineDist},
        {"stat_chebyshev_dist", cc::ChebyshevDist},
        // Two-sample effect sizes (double)
        {"stat_cohens_d2", cc::CohensD2},
        {"stat_hedges_g2", cc::HedgesG2},
        {"stat_glass_delta", cc::GlassDelta},
    };
    return table;
}

/// Parameterized two-column aggregates: (LIST, LIST, DOUBLE) -> DOUBLE (2 functions).
inline const std::vector<std::pair<std::string, TwoListParamFn>>& TwoListParamFunctionTable() {
    static const std::vector<std::pair<std::string, TwoListParamFn>> table = {
        {"stat_weighted_percentile", cc::WeightedPercentile},
        {"stat_minkowski_dist", cc::MinkowskiDist},
    };
    return table;
}

/// Window functions: (LIST[, DOUBLE]) -> LIST (23 functions).
/// pair: {name, {compute function, has_param}}
inline const std::vector<std::pair<std::string, std::pair<WindowFn, bool>>>& WindowFunctionTable() {
    static const std::vector<std::pair<std::string, std::pair<WindowFn, bool>>> table = {
        // Rolling statistics (window size)
        {"stat_rolling_mean", {cc::WfRollingMean, true}},
        {"stat_rolling_std", {cc::WfRollingStd, true}},
        {"stat_rolling_min", {cc::WfRollingMin, true}},
        {"stat_rolling_max", {cc::WfRollingMax, true}},
        {"stat_rolling_sum", {cc::WfRollingSum, true}},
        // Moving averages
        {"stat_moving_avg", {cc::WfMovingAvg, true}},
        {"stat_ema", {cc::WfEma, true}},
        // Rank (no parameter)
        {"stat_rank", {cc::WfRank, false}},
        // Missing-value fill (no parameter)
        {"stat_fillna_mean", {cc::WfFillnaMean, false}},
        {"stat_fillna_median", {cc::WfFillnaMedian, false}},
        {"stat_fillna_ffill", {cc::WfFillnaFfill, false}},
        {"stat_fillna_bfill", {cc::WfFillnaBfill, false}},
        {"stat_fillna_interp", {cc::WfFillnaInterp, false}},
        // Encoding / binning
        {"stat_label_encode", {cc::WfLabelEncode, false}},
        {"stat_bin_width", {cc::WfBinWidth, true}},
        {"stat_bin_freq", {cc::WfBinFreq, true}},
        // Time series (lag / diff / seasonal diff)
        {"stat_lag", {cc::WfLag, true}},
        {"stat_diff", {cc::WfDiff, true}},
        {"stat_seasonal_diff", {cc::WfSeasonalDiff, true}},
        // Outlier detection (no parameter)
        {"stat_outliers_iqr", {cc::WfOutliersIqr, false}},
        {"stat_outliers_zscore", {cc::WfOutliersZscore, false}},
        {"stat_outliers_mzscore", {cc::WfOutliersMzscore, false}},
        // Robust
        {"stat_winsorize", {cc::WfWinsorize, false}},
    };
    return table;
}

// ---------------------------------------------------------------------------
// Public API
//   - LIST-based registration is RegisterStatcppListFunctions below.
//   - Scalar registration and the overall umbrella (RegisterStatcppFunctions)
//     are defined in statcpp_scalar.hpp.
// ---------------------------------------------------------------------------

/**
 * @brief Register the LIST-based functions (aggregate / two-column / window) and extras.
 */
inline void RegisterStatcppListFunctions(duckdb::Connection& con) {
    for (const auto& [name, fn] : ScalarFunctionTable()) {
        RegisterListToScalar(con, name, fn);
    }
    for (const auto& [name, fn_sort] : ParamScalarFunctionTable()) {
        RegisterListParamToScalar(con, name, fn_sort.first, fn_sort.second);
    }
    for (const auto& [name, fn] : TwoListFunctionTable()) {
        RegisterTwoListToScalar(con, name, fn);
    }
    for (const auto& [name, fn] : TwoListParamFunctionTable()) {
        RegisterTwoListParamToScalar(con, name, fn);
    }
    for (const auto& [name, fn_param] : WindowFunctionTable()) {
        RegisterWindow(con, name, fn_param.first, fn_param.second);
    }

    // --- DuckDB-only extras (convenience functions with no sqlite3-stats equivalent) ---

    // Sum / count / min / max (offered on a LIST, distinct from the SQL-native ones).
    RegisterListToScalar(con, "stat_sum",
                         [](const Vec& v) { return statcpp::sum(v.begin(), v.end()); });
    RegisterListToScalar(con, "stat_count", [](const Vec& v) {
        return static_cast<double>(statcpp::count(v.begin(), v.end()));
    });
    RegisterListToScalar(con, "stat_minimum", [](const Vec& v) {
        return static_cast<double>(statcpp::minimum(v.begin(), v.end()));
    });
    RegisterListToScalar(con, "stat_maximum", [](const Vec& v) {
        return static_cast<double>(statcpp::maximum(v.begin(), v.end()));
    });

    // Math transforms (column -> column).
    RegisterListToList(con, "stat_log_transform", [](const Vec& v) { return statcpp::log_transform(v); });
    RegisterListToList(con, "stat_sqrt_transform", [](const Vec& v) { return statcpp::sqrt_transform(v); });

    // Missing rate (needs the raw order, so drop_na = false).
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

    // Mean imputation (fill each missing value with the observed mean).
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
