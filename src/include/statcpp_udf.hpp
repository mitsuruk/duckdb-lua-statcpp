/**
 * @file statcpp_udf.hpp
 * @brief statcpp 関数を DuckDB SQL UDF として登録する(集約 / 2 列 / Window / 独自拡張)
 *
 * 設計
 * ----
 * 全体を保持して計算する統計量(median, MAD, percentile, correlation, ...)は
 * DuckDB の固定長 POD 集約ステートに収まらない. そこで CreateAggregateFunction では
 * なく次のパターンを用いる:
 *
 *   list() で列を LIST に集約 → LIST を受け取る vectorized scalar UDF が
 *   std::vector<double> に展開し, statcpp の vector / iterator API へ転送する.
 *
 * 登録はテーブル駆動: 各関数は {SQL 名, 計算関数} の 1 エントリで表現する.
 * 計算本体は statcpp_compute.hpp の純粋関数(compute::*)に分離している.
 *
 * UDF 境界での規約
 * ----------------
 *  - DuckDB NULL <-> statcpp 欠損値(NaN). LIST 要素の NULL は NaN として入力され,
 *    結果要素の NaN は SQL NULL として出力される.
 *  - 単標本 / 2 標本統計では計算前に欠損値を除去する(2 標本は対応位置でペア除去).
 *  - statcpp は不正入力で例外を投げうるため, 全 UDF は境界で catch し SQL NULL を返す.
 *
 * スカラ関数(分布・検定補助)は statcpp_scalar.hpp で別途登録する.
 *
 * 各統計量の意味・前提・アルゴリズムは statcpp のドキュメントを参照:
 * https://github.com/mitsuruk/statcpp(本層は SQL への公開のみを担う).
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

/// statcpp へ渡すサンプルベクタ型.
using Vec = std::vector<double>;

// ---------------------------------------------------------------------------
// 型ヘルパ
// ---------------------------------------------------------------------------

/// LIST<DOUBLE> 論理型を生成する.
inline duckdb::LogicalType ListOfDouble() {
    return duckdb::LogicalType::LIST(duckdb::LogicalType::DOUBLE);
}

/**
 * @brief DuckDB の LIST 値を std::vector<double> に変換する.
 *
 * NULL 要素は NaN(DuckDB NULL -> statcpp 欠損値)に変換する.
 * 値そのものが NULL の場合は空ベクタを返す.
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

/// 欠損値(NaN)を除いたコピーを返す.
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

/// ベクタから LIST<DOUBLE> 値を生成する. NaN 要素は SQL NULL になる.
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
// UDF シグネチャ(プレーンなベクタを扱う. 境界 glue は下記の登録ヘルパ)
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
// 登録ヘルパ
// ---------------------------------------------------------------------------

/**
 * @brief LIST<DOUBLE> -> DOUBLE の UDF を登録する.
 * @param drop_na true の場合, 欠損値を除去し昇順ソートしてから fn を呼ぶ.
 *        基本統計量は順序非依存なのでソートは無害, かつ median/iqr 等の前処理になる.
 *        欠損率など生の並びが必要なものは false を指定する.
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
 * @brief (LIST<DOUBLE>, DOUBLE) -> DOUBLE の UDF を登録する.
 * @param sort true の場合, 欠損除去後に昇順ソートする(percentile/trimmed_mean 等).
 *        時系列(autocorrelation 等)は順序依存のため false を指定する.
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

/// 対応位置で 2 つの LIST をペア化し, 欠損ペアを除いた 2 ベクタを得る.
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

/// (LIST<DOUBLE>, LIST<DOUBLE>) -> DOUBLE の UDF を登録する(欠損ペア除去).
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

/// (LIST<DOUBLE>, LIST<DOUBLE>, DOUBLE) -> DOUBLE の UDF を登録する(欠損ペア除去).
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
 * @brief LIST<DOUBLE> -> LIST<DOUBLE> の列変換 UDF を登録する.
 *
 * 生のベクタ(欠損は NaN のまま)を fn に渡す. 結果の NaN 要素は SQL NULL になる.
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
 * @brief Window 系 UDF を登録する.
 *
 * sqlite3 版の真の Window 関数を, DuckDB では list() で集約した LIST を受け取り
 * 同じ長さの LIST を返す変換として表現する(unnest で行へ展開).
 * 入力 LIST の NULL 要素は NaN として欠損フラグ化し fn に渡す.
 * @param has_param true で 2 番目の DOUBLE 引数(窓幅・ラグ等)を int として渡す.
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
// 関数テーブル
// ---------------------------------------------------------------------------

namespace cc = statcpp_duckdb::compute;

/// 基本集約: LIST<DOUBLE> -> DOUBLE(24 関数, 欠損除去 + ソート済みで呼ばれる).
inline const std::vector<std::pair<std::string, ScalarFn>>& ScalarFunctionTable() {
    static const std::vector<std::pair<std::string, ScalarFn>> table = {
        // 基本統計量
        {"stat_mean", cc::Mean},
        {"stat_median", cc::Median},
        {"stat_mode", cc::Mode},
        {"stat_geometric_mean", cc::GeometricMean},
        {"stat_harmonic_mean", cc::HarmonicMean},
        // ばらつき / 散布度
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
        // 分布の形状
        {"stat_population_skewness", cc::PopulationSkewness},
        {"stat_skewness", cc::Skewness},
        {"stat_population_kurtosis", cc::PopulationKurtosis},
        {"stat_kurtosis", cc::Kurtosis},
        // 推定
        {"stat_se", cc::Se},
        // ロバスト統計
        {"stat_mad", cc::Mad},
        {"stat_mad_scaled", cc::MadScaled},
        {"stat_hodges_lehmann", cc::HodgesLehmann},
    };
    return table;
}

/// パラメータ集約: (LIST<DOUBLE>, DOUBLE) -> DOUBLE(7 関数).
/// pair: {名前, {計算関数, ソート要否}}
inline const std::vector<std::pair<std::string, std::pair<ParamScalarFn, bool>>>&
ParamScalarFunctionTable() {
    static const std::vector<std::pair<std::string, std::pair<ParamScalarFn, bool>>> table = {
        {"stat_trimmed_mean", {cc::TrimmedMean, true}},
        {"stat_percentile", {cc::Percentile, true}},
        {"stat_moe_mean", {cc::MoeMean, true}},
        {"stat_cohens_d", {cc::CohensD, true}},
        {"stat_hedges_g", {cc::HedgesG, true}},
        {"stat_acf_lag", {cc::AcfLag, false}},  // 時系列: 順序を保持
        {"stat_biweight_midvar", {cc::BiweightMidvar, true}},
    };
    return table;
}

/// 2 列集約: (LIST, LIST) -> DOUBLE(22 関数 + 効果量 3 関数).
inline const std::vector<std::pair<std::string, TwoListFn>>& TwoListFunctionTable() {
    static const std::vector<std::pair<std::string, TwoListFn>> table = {
        // 相関 / 共分散
        {"stat_population_covariance", cc::PopulationCovariance},
        {"stat_covariance", cc::Covariance},
        {"stat_pearson_r", cc::PearsonR},
        {"stat_spearman_r", cc::SpearmanR},
        {"stat_kendall_tau", cc::KendallTau},
        {"stat_weighted_covariance", cc::WeightedCovariance},
        // 重み付き統計
        {"stat_weighted_mean", cc::WeightedMean},
        {"stat_weighted_harmonic_mean", cc::WeightedHarmonicMean},
        {"stat_weighted_variance", cc::WeightedVariance},
        {"stat_weighted_stddev", cc::WeightedStddev},
        {"stat_weighted_median", cc::WeightedMedian},
        // 回帰(double)
        {"stat_r_squared", cc::RSquared},
        {"stat_adjusted_r_squared", cc::AdjustedRSquared},
        // 誤差メトリクス
        {"stat_mae", cc::Mae},
        {"stat_mse", cc::Mse},
        {"stat_rmse", cc::Rmse},
        {"stat_mape", cc::Mape},
        // 距離メトリクス
        {"stat_euclidean_dist", cc::EuclideanDist},
        {"stat_manhattan_dist", cc::ManhattanDist},
        {"stat_cosine_sim", cc::CosineSim},
        {"stat_cosine_dist", cc::CosineDist},
        {"stat_chebyshev_dist", cc::ChebyshevDist},
        // 2 標本効果量(double)
        {"stat_cohens_d2", cc::CohensD2},
        {"stat_hedges_g2", cc::HedgesG2},
        {"stat_glass_delta", cc::GlassDelta},
    };
    return table;
}

/// パラメータ付き 2 列集約: (LIST, LIST, DOUBLE) -> DOUBLE(2 関数).
inline const std::vector<std::pair<std::string, TwoListParamFn>>& TwoListParamFunctionTable() {
    static const std::vector<std::pair<std::string, TwoListParamFn>> table = {
        {"stat_weighted_percentile", cc::WeightedPercentile},
        {"stat_minkowski_dist", cc::MinkowskiDist},
    };
    return table;
}

/// Window 系: (LIST[, DOUBLE]) -> LIST(23 関数).
/// pair: {名前, {計算関数, パラメータ有無}}
inline const std::vector<std::pair<std::string, std::pair<WindowFn, bool>>>& WindowFunctionTable() {
    static const std::vector<std::pair<std::string, std::pair<WindowFn, bool>>> table = {
        // ローリング統計(窓幅指定)
        {"stat_rolling_mean", {cc::WfRollingMean, true}},
        {"stat_rolling_std", {cc::WfRollingStd, true}},
        {"stat_rolling_min", {cc::WfRollingMin, true}},
        {"stat_rolling_max", {cc::WfRollingMax, true}},
        {"stat_rolling_sum", {cc::WfRollingSum, true}},
        // 移動平均
        {"stat_moving_avg", {cc::WfMovingAvg, true}},
        {"stat_ema", {cc::WfEma, true}},
        // ランク(引数なし)
        {"stat_rank", {cc::WfRank, false}},
        // 欠損補完(引数なし)
        {"stat_fillna_mean", {cc::WfFillnaMean, false}},
        {"stat_fillna_median", {cc::WfFillnaMedian, false}},
        {"stat_fillna_ffill", {cc::WfFillnaFfill, false}},
        {"stat_fillna_bfill", {cc::WfFillnaBfill, false}},
        {"stat_fillna_interp", {cc::WfFillnaInterp, false}},
        // エンコード / ビニング
        {"stat_label_encode", {cc::WfLabelEncode, false}},
        {"stat_bin_width", {cc::WfBinWidth, true}},
        {"stat_bin_freq", {cc::WfBinFreq, true}},
        // 時系列(ラグ / 階差 / 季節階差)
        {"stat_lag", {cc::WfLag, true}},
        {"stat_diff", {cc::WfDiff, true}},
        {"stat_seasonal_diff", {cc::WfSeasonalDiff, true}},
        // 外れ値検出(引数なし)
        {"stat_outliers_iqr", {cc::WfOutliersIqr, false}},
        {"stat_outliers_zscore", {cc::WfOutliersZscore, false}},
        {"stat_outliers_mzscore", {cc::WfOutliersMzscore, false}},
        // ロバスト
        {"stat_winsorize", {cc::WfWinsorize, false}},
    };
    return table;
}

// ---------------------------------------------------------------------------
// 公開 API
//   - LIST 系の登録は下記 RegisterStatcppListFunctions
//   - スカラ系の登録と全体の umbrella(RegisterStatcppFunctions)は
//     statcpp_scalar.hpp 側で定義する.
// ---------------------------------------------------------------------------

/**
 * @brief LIST 系(集約 / 2 列 / Window)と独自拡張を登録する.
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

    // --- DuckDB 独自拡張(sqlite3 版に対応関数が無い便利関数) ---

    // 総和 / 件数 / 最小 / 最大(SQL ネイティブとは別に LIST に対して提供).
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

    // 数学変換(列 -> 列).
    RegisterListToList(con, "stat_log_transform", [](const Vec& v) { return statcpp::log_transform(v); });
    RegisterListToList(con, "stat_sqrt_transform", [](const Vec& v) { return statcpp::sqrt_transform(v); });

    // 欠損率(生の並びが必要なため drop_na = false).
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

    // 平均値補完(各欠損を観測値の平均で埋める).
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
