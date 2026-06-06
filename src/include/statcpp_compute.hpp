/**
 * @file statcpp_compute.hpp
 * @brief Pure compute functions over statcpp (database-agnostic)
 *
 * Ported from the `calc_*` / `wf_*` wrappers of sqlite3StatisticalLibrary. None of
 * these depend on DuckDB or SQLite; they take and return `std::vector<double>`.
 * The marshalling layers in statcpp_udf.hpp / statcpp_scalar.hpp call into them.
 *
 * Conventions
 * -----------
 *  - Aggregates: `(const Vec&[, const Vec&][, double]) -> double`
 *  - Window functions: `(const Vec& values, const std::vector<bool>& nulls, int param) -> Vec`
 *    `values` keeps missing entries as NaN; `nulls[i]` flags a missing entry.
 *  - NaN / Inf results are converted to SQL NULL by the marshalling layer.
 *
 * For the definition, assumptions and algorithm of each statistic, see the statcpp
 * documentation.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <vector>

#include "statcpp/statcpp.hpp"

namespace statcpp_duckdb::compute {

/// Sample vector type passed to statcpp.
using Vec = std::vector<double>;

/// Shorthand for a NaN result (representation of invalid input).
inline double NaN() {
    return std::numeric_limits<double>::quiet_NaN();
}

// ===========================================================================
// Basic aggregates: (const Vec&) -> double
// The input is already NaN-free (and sorted, where the helper requests it).
// ===========================================================================

// --- Basic statistics ---

inline double Mean(const Vec& v) { return statcpp::mean(v.begin(), v.end()); }

inline double Median(const Vec& v) {
    Vec s(v);
    std::sort(s.begin(), s.end());
    return statcpp::median(s.begin(), s.end());
}

inline double Mode(const Vec& v) { return statcpp::mode(v.begin(), v.end()); }

inline double GeometricMean(const Vec& v) { return statcpp::geometric_mean(v.begin(), v.end()); }

inline double HarmonicMean(const Vec& v) { return statcpp::harmonic_mean(v.begin(), v.end()); }

// --- Dispersion / spread ---

inline double Range(const Vec& v) { return statcpp::range(v.begin(), v.end()); }

inline double Var(const Vec& v) {
    // stat_var defaults to ddof=0 (same as the population variance).
    return statcpp::var(v.begin(), v.end(), static_cast<std::size_t>(0));
}

inline double PopulationVariance(const Vec& v) { return statcpp::population_variance(v.begin(), v.end()); }

inline double SampleVariance(const Vec& v) {
    if (v.size() < 2) return NaN();
    return statcpp::sample_variance(v.begin(), v.end());
}

inline double Stdev(const Vec& v) {
    return statcpp::stdev(v.begin(), v.end(), static_cast<std::size_t>(0));
}

inline double PopulationStddev(const Vec& v) { return statcpp::population_stddev(v.begin(), v.end()); }

inline double SampleStddev(const Vec& v) {
    if (v.size() < 2) return NaN();
    return statcpp::sample_stddev(v.begin(), v.end());
}

inline double Cv(const Vec& v) { return statcpp::coefficient_of_variation(v.begin(), v.end()); }

inline double Iqr(const Vec& v) {
    Vec s(v);
    std::sort(s.begin(), s.end());
    return statcpp::iqr(s.begin(), s.end());
}

inline double MadMean(const Vec& v) { return statcpp::mean_absolute_deviation(v.begin(), v.end()); }

inline double GeometricStddev(const Vec& v) { return statcpp::geometric_stddev(v.begin(), v.end()); }

// --- Shape of distribution ---

inline double PopulationSkewness(const Vec& v) {
    if (v.size() < 3) return NaN();
    return statcpp::population_skewness(v.begin(), v.end());
}

inline double Skewness(const Vec& v) {
    if (v.size() < 3) return NaN();
    return statcpp::sample_skewness(v.begin(), v.end());
}

inline double PopulationKurtosis(const Vec& v) {
    if (v.size() < 4) return NaN();
    return statcpp::population_kurtosis(v.begin(), v.end());
}

inline double Kurtosis(const Vec& v) {
    if (v.size() < 4) return NaN();
    return statcpp::sample_kurtosis(v.begin(), v.end());
}

// --- Estimation ---

inline double Se(const Vec& v) {
    if (v.size() < 2) return NaN();
    return statcpp::standard_error(v.begin(), v.end());
}

// --- Robust statistics ---

inline double Mad(const Vec& v) { return statcpp::mad(v.begin(), v.end()); }

inline double MadScaled(const Vec& v) { return statcpp::mad_scaled(v.begin(), v.end()); }

inline double HodgesLehmann(const Vec& v) { return statcpp::hodges_lehmann(v.begin(), v.end()); }

// ===========================================================================
// Parameterized aggregates: (const Vec&, double) -> double
// ===========================================================================

inline double TrimmedMean(const Vec& v, double proportion) {
    Vec s(v);
    std::sort(s.begin(), s.end());
    return statcpp::trimmed_mean(s.begin(), s.end(), proportion);
}

inline double Percentile(const Vec& v, double p) {
    Vec s(v);
    std::sort(s.begin(), s.end());
    return statcpp::percentile(s.begin(), s.end(), p);
}

inline double MoeMean(const Vec& v, double conf) {
    if (v.size() < 2) return NaN();
    return statcpp::margin_of_error_mean(v.begin(), v.end(), conf);
}

inline double CohensD(const Vec& v, double mu0) {
    if (v.size() < 2) return NaN();
    return statcpp::cohens_d(v.begin(), v.end(), mu0);
}

inline double HedgesG(const Vec& v, double mu0) {
    if (v.size() < 2) return NaN();
    return statcpp::hedges_g(v.begin(), v.end(), mu0);
}

inline double AcfLag(const Vec& v, double lag_d) {
    auto lag = static_cast<std::size_t>(lag_d);
    if (lag >= v.size()) return NaN();
    return statcpp::autocorrelation(v.begin(), v.end(), lag);
}

inline double BiweightMidvar(const Vec& v, double c) {
    return statcpp::biweight_midvariance(v.begin(), v.end(), c);
}

// ===========================================================================
// Two-column aggregates: (const Vec&, const Vec&) -> double
// Inputs are already pairwise-aligned (missing pairs dropped by the marshalling layer).
// ===========================================================================

// --- Correlation / covariance ---

inline double PopulationCovariance(const Vec& x, const Vec& y) {
    if (x.size() < 2) return NaN();
    return statcpp::population_covariance(x.begin(), x.end(), y.begin(), y.end());
}

inline double Covariance(const Vec& x, const Vec& y) {
    if (x.size() < 2) return NaN();
    return statcpp::covariance(x.begin(), x.end(), y.begin(), y.end());
}

inline double PearsonR(const Vec& x, const Vec& y) {
    if (x.size() < 2) return NaN();
    return statcpp::pearson_correlation(x.begin(), x.end(), y.begin(), y.end());
}

inline double SpearmanR(const Vec& x, const Vec& y) {
    if (x.size() < 2) return NaN();
    return statcpp::spearman_correlation(x.begin(), x.end(), y.begin(), y.end());
}

inline double KendallTau(const Vec& x, const Vec& y) {
    if (x.size() < 2) return NaN();
    return statcpp::kendall_tau(x.begin(), x.end(), y.begin(), y.end());
}

inline double WeightedCovariance(const Vec& values, const Vec& weights) {
    // With a 2-column interface this is the weighted self-covariance of the value
    // column (i.e. the weighted variance).
    if (values.size() < 2) return NaN();
    return statcpp::weighted_covariance(values.begin(), values.end(),
                                        values.begin(), values.end(), weights.begin());
}

// --- Weighted statistics ---

inline double WeightedMean(const Vec& values, const Vec& weights) {
    return statcpp::weighted_mean(values.begin(), values.end(), weights.begin(), weights.end());
}

inline double WeightedHarmonicMean(const Vec& values, const Vec& weights) {
    return statcpp::weighted_harmonic_mean(values.begin(), values.end(), weights.begin(), weights.end());
}

inline double WeightedVariance(const Vec& values, const Vec& weights) {
    return statcpp::weighted_variance(values.begin(), values.end(), weights.begin(), weights.end());
}

inline double WeightedStddev(const Vec& values, const Vec& weights) {
    return statcpp::weighted_stddev(values.begin(), values.end(), weights.begin(), weights.end());
}

inline double WeightedMedian(const Vec& values, const Vec& weights) {
    return statcpp::weighted_median(values.begin(), values.end(), weights.begin(), weights.end());
}

inline double WeightedPercentile(const Vec& values, const Vec& weights, double p) {
    return statcpp::weighted_percentile(values.begin(), values.end(),
                                        weights.begin(), weights.end(), p);
}

// --- Regression (the double-returning ones only) ---

inline double RSquared(const Vec& actual, const Vec& predicted) {
    if (actual.size() < 2) return NaN();
    return statcpp::r_squared(actual.begin(), actual.end(), predicted.begin(), predicted.end());
}

inline double AdjustedRSquared(const Vec& actual, const Vec& predicted) {
    if (actual.size() < 3) return NaN();
    return statcpp::adjusted_r_squared(actual.begin(), actual.end(),
                                       predicted.begin(), predicted.end(), 1);
}

// --- Error metrics ---

inline double Mae(const Vec& actual, const Vec& predicted) {
    return statcpp::mae(actual.begin(), actual.end(), predicted.begin());
}

inline double Mse(const Vec& actual, const Vec& predicted) {
    return statcpp::mse(actual.begin(), actual.end(), predicted.begin());
}

inline double Rmse(const Vec& actual, const Vec& predicted) {
    return statcpp::rmse(actual.begin(), actual.end(), predicted.begin());
}

inline double Mape(const Vec& actual, const Vec& predicted) {
    return statcpp::mape(actual.begin(), actual.end(), predicted.begin());
}

// --- Distance metrics ---

inline double EuclideanDist(const Vec& a, const Vec& b) {
    return statcpp::euclidean_distance(a.begin(), a.end(), b.begin(), b.end());
}

inline double ManhattanDist(const Vec& a, const Vec& b) {
    return statcpp::manhattan_distance(a.begin(), a.end(), b.begin(), b.end());
}

inline double CosineSim(const Vec& a, const Vec& b) {
    return statcpp::cosine_similarity(a.begin(), a.end(), b.begin(), b.end());
}

inline double CosineDist(const Vec& a, const Vec& b) {
    return statcpp::cosine_distance(a.begin(), a.end(), b.begin(), b.end());
}

inline double MinkowskiDist(const Vec& a, const Vec& b, double p) {
    return statcpp::minkowski_distance(a.begin(), a.end(), b.begin(), b.end(), p);
}

inline double ChebyshevDist(const Vec& a, const Vec& b) {
    return statcpp::chebyshev_distance(a.begin(), a.end(), b.begin(), b.end());
}

// --- Two-sample effect sizes (the double-returning ones) ---

inline double CohensD2(const Vec& x, const Vec& y) {
    if (x.size() < 2 || y.size() < 2) return NaN();
    return statcpp::cohens_d_two_sample(x.begin(), x.end(), y.begin(), y.end());
}

inline double HedgesG2(const Vec& x, const Vec& y) {
    if (x.size() < 2 || y.size() < 2) return NaN();
    return statcpp::hedges_g_two_sample(x.begin(), x.end(), y.begin(), y.end());
}

inline double GlassDelta(const Vec& x, const Vec& y) {
    if (x.size() < 2 || y.size() < 2) return NaN();
    return statcpp::glass_delta(x.begin(), x.end(), y.begin(), y.end());
}

// ===========================================================================
// Window functions: (const Vec& values, const std::vector<bool>& nulls, int param) -> Vec
// `values` keeps missing entries as NaN. The result has the same length (missing
// positions are NaN).
// ===========================================================================

/// Extract the non-missing values (for statcpp functions that do not handle NaN).
inline Vec ExtractValid(const Vec& values, const std::vector<bool>& nulls) {
    Vec valid;
    valid.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!nulls[i]) valid.push_back(values[i]);
    }
    return valid;
}

inline Vec WfRollingMean(const Vec& values, const std::vector<bool>& nulls, int window) {
    if (window <= 0) window = 1;
    auto result = statcpp::rolling_mean(values, static_cast<std::size_t>(window));
    for (std::size_t i = 0; i < result.size() && i < nulls.size(); ++i) {
        if (nulls[i]) result[i] = NaN();
    }
    return result;
}

inline Vec WfRollingStd(const Vec& values, const std::vector<bool>& nulls, int window) {
    if (window <= 0) window = 1;
    auto result = statcpp::rolling_std(values, static_cast<std::size_t>(window));
    for (std::size_t i = 0; i < result.size() && i < nulls.size(); ++i) {
        if (nulls[i]) result[i] = NaN();
    }
    return result;
}

inline Vec WfRollingMin(const Vec& values, const std::vector<bool>& nulls, int window) {
    if (window <= 0) window = 1;
    auto result = statcpp::rolling_min(values, static_cast<std::size_t>(window));
    for (std::size_t i = 0; i < result.size() && i < nulls.size(); ++i) {
        if (nulls[i]) result[i] = NaN();
    }
    return result;
}

inline Vec WfRollingMax(const Vec& values, const std::vector<bool>& nulls, int window) {
    if (window <= 0) window = 1;
    auto result = statcpp::rolling_max(values, static_cast<std::size_t>(window));
    for (std::size_t i = 0; i < result.size() && i < nulls.size(); ++i) {
        if (nulls[i]) result[i] = NaN();
    }
    return result;
}

inline Vec WfRollingSum(const Vec& values, const std::vector<bool>& nulls, int window) {
    if (window <= 0) window = 1;
    auto result = statcpp::rolling_sum(values, static_cast<std::size_t>(window));
    for (std::size_t i = 0; i < result.size() && i < nulls.size(); ++i) {
        if (nulls[i]) result[i] = NaN();
    }
    return result;
}

inline Vec WfMovingAvg(const Vec& values, const std::vector<bool>& /*nulls*/, int window) {
    if (window <= 0) window = 1;
    return statcpp::moving_average(values.begin(), values.end(), static_cast<std::size_t>(window));
}

inline Vec WfEma(const Vec& values, const std::vector<bool>& /*nulls*/, int span) {
    if (span <= 0) span = 10;
    const double alpha = 2.0 / (static_cast<double>(span) + 1.0);
    return statcpp::exponential_moving_average(values.begin(), values.end(), alpha);
}

inline Vec WfRank(const Vec& values, const std::vector<bool>& nulls, int /*param*/) {
    auto valid = ExtractValid(values, nulls);
    if (valid.empty()) return Vec(values.size(), NaN());
    auto ranks = statcpp::rank_transform(valid);
    Vec result(values.size(), NaN());
    std::size_t vi = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!nulls[i]) result[i] = ranks[vi++];
    }
    return result;
}

inline Vec WfFillnaMean(const Vec& values, const std::vector<bool>& /*nulls*/, int /*param*/) {
    return statcpp::fillna_mean(values);
}

inline Vec WfFillnaMedian(const Vec& values, const std::vector<bool>& /*nulls*/, int /*param*/) {
    return statcpp::fillna_median(values);
}

inline Vec WfFillnaFfill(const Vec& values, const std::vector<bool>& /*nulls*/, int /*param*/) {
    return statcpp::fillna_ffill(values);
}

inline Vec WfFillnaBfill(const Vec& values, const std::vector<bool>& /*nulls*/, int /*param*/) {
    return statcpp::fillna_bfill(values);
}

inline Vec WfFillnaInterp(const Vec& values, const std::vector<bool>& /*nulls*/, int /*param*/) {
    return statcpp::fillna_interpolate(values);
}

inline Vec WfLabelEncode(const Vec& values, const std::vector<bool>& nulls, int /*param*/) {
    auto valid = ExtractValid(values, nulls);
    if (valid.empty()) return Vec(values.size(), NaN());
    auto enc = statcpp::label_encode(valid);
    Vec result(values.size(), NaN());
    std::size_t vi = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!nulls[i]) result[i] = static_cast<double>(enc.encoded[vi++]);
    }
    return result;
}

inline Vec WfBinWidth(const Vec& values, const std::vector<bool>& nulls, int n_bins) {
    if (n_bins <= 0) n_bins = 10;
    auto valid = ExtractValid(values, nulls);
    if (valid.empty()) return Vec(values.size(), NaN());
    auto bins = statcpp::bin_equal_width(valid, static_cast<std::size_t>(n_bins));
    Vec result(values.size(), NaN());
    std::size_t vi = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!nulls[i]) result[i] = static_cast<double>(bins[vi++]);
    }
    return result;
}

inline Vec WfBinFreq(const Vec& values, const std::vector<bool>& nulls, int n_bins) {
    if (n_bins <= 0) n_bins = 10;
    auto valid = ExtractValid(values, nulls);
    if (valid.empty()) return Vec(values.size(), NaN());
    auto bins = statcpp::bin_equal_freq(valid, static_cast<std::size_t>(n_bins));
    Vec result(values.size(), NaN());
    std::size_t vi = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!nulls[i]) result[i] = static_cast<double>(bins[vi++]);
    }
    return result;
}

inline Vec WfLag(const Vec& values, const std::vector<bool>& /*nulls*/, int k) {
    if (k <= 0) k = 1;
    return statcpp::lag(values.begin(), values.end(), static_cast<std::size_t>(k));
}

inline Vec WfDiff(const Vec& values, const std::vector<bool>& /*nulls*/, int order) {
    if (order <= 0) order = 1;
    auto result = statcpp::diff(values.begin(), values.end(), static_cast<std::size_t>(order));
    Vec padded(values.size(), NaN());
    for (std::size_t i = 0; i < result.size(); ++i) {
        padded[i + static_cast<std::size_t>(order)] = result[i];
    }
    return padded;
}

inline Vec WfSeasonalDiff(const Vec& values, const std::vector<bool>& /*nulls*/, int period) {
    if (period <= 0) period = 1;
    auto result = statcpp::seasonal_diff(values.begin(), values.end(), static_cast<std::size_t>(period));
    Vec padded(values.size(), NaN());
    for (std::size_t i = 0; i < result.size(); ++i) {
        padded[i + static_cast<std::size_t>(period)] = result[i];
    }
    return padded;
}

/// Shared outlier helper: run a detector on the valid values and emit 1.0 (outlier) / 0.0 / NaN.
template <typename Detector>
inline Vec OutlierFlags(const Vec& values, const std::vector<bool>& nulls, Detector detect) {
    auto valid = ExtractValid(values, nulls);
    if (valid.empty()) return Vec(values.size(), NaN());
    auto det = detect(valid.begin(), valid.end());
    std::vector<bool> is_outlier(valid.size(), false);
    for (auto idx : det.outlier_indices) {
        if (idx < is_outlier.size()) is_outlier[idx] = true;
    }
    Vec result(values.size(), NaN());
    std::size_t vi = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!nulls[i]) result[i] = is_outlier[vi++] ? 1.0 : 0.0;
    }
    return result;
}

inline Vec WfOutliersIqr(const Vec& values, const std::vector<bool>& nulls, int /*param*/) {
    return OutlierFlags(values, nulls, [](auto f, auto l) { return statcpp::detect_outliers_iqr(f, l); });
}

inline Vec WfOutliersZscore(const Vec& values, const std::vector<bool>& nulls, int /*param*/) {
    return OutlierFlags(values, nulls, [](auto f, auto l) { return statcpp::detect_outliers_zscore(f, l); });
}

inline Vec WfOutliersMzscore(const Vec& values, const std::vector<bool>& nulls, int /*param*/) {
    return OutlierFlags(values, nulls,
                        [](auto f, auto l) { return statcpp::detect_outliers_modified_zscore(f, l); });
}

inline Vec WfWinsorize(const Vec& values, const std::vector<bool>& nulls, int /*param*/) {
    auto valid = ExtractValid(values, nulls);
    if (valid.empty()) return Vec(values.size(), NaN());
    auto wins = statcpp::winsorize(valid.begin(), valid.end());
    Vec result(values.size(), NaN());
    std::size_t vi = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!nulls[i]) result[i] = wins[vi++];
    }
    return result;
}

}  // namespace statcpp_duckdb::compute
