/**
 * @file statcpp_scalar.hpp
 * @brief Register statcpp scalar functions (distributions / special functions / effect sizes / ...) as DuckDB UDFs
 *
 * These take a fixed number of DOUBLE arguments and return DOUBLE (a few return
 * VARCHAR), as plain scalar functions. Ported from the `sf_*` wrappers of
 * sqlite3StatisticalLibrary.
 *
 * Design (PoC simplification)
 * ---------------------------
 *  - Every function is registered with a fixed arity (all arguments required). The
 *    variadic / defaulted arguments of the sqlite3 version are not replicated; all
 *    arguments are passed explicitly (avoids DuckDB overload / vararg complexity).
 *  - If any argument is NULL, the result is SQL NULL. A NaN/Inf result is NULL too.
 *  - statcpp functions that take integers receive the DOUBLE arguments cast to the
 *    appropriate integer type.
 *
 * Random functions (*_rand)
 * -------------------------
 *  DuckDB's CreateVectorizedFunction exposes no volatility (non-deterministic) flag,
 *  so these are registered as deterministic functions. A call with only constant
 *  arguments may be constant-folded to a single value across all rows (a known PoC
 *  limitation). Pass a column argument to evaluate per row.
 *
 * JSON-returning functions (test-result / confidence-interval objects, ...) are out of
 * scope for this PoC.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "duckdb.hpp"

#include "statcpp/statcpp.hpp"
#include "statcpp_udf.hpp"

namespace statcpp_duckdb {

/// Fixed-arity scalar compute function: argument vector (size == arity) -> double.
using ScalarMathFn = std::function<double(const std::vector<double>&)>;
/// Fixed-arity scalar compute function: argument vector -> string (label, ...).
using ScalarStringFn = std::function<std::string(const std::vector<double>&)>;

// ---------------------------------------------------------------------------
// Scalar registration helpers
// ---------------------------------------------------------------------------

/**
 * @brief Register a scalar UDF (DOUBLE x arity) -> DOUBLE.
 *
 * Returns NULL if any argument is NULL. A NaN/Inf result is NULL too.
 */
inline void RegisterScalarMath(duckdb::Connection& con, const std::string& name, std::size_t arity,
                               ScalarMathFn fn) {
    duckdb::scalar_function_t udf =
        [fn, arity](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
                    duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            std::vector<double> a;
            a.reserve(arity);
            bool any_null = false;
            for (std::size_t j = 0; j < arity; ++j) {
                const duckdb::Value v = args.data[j].GetValue(i);
                if (v.IsNull()) {
                    any_null = true;
                    break;
                }
                a.push_back(v.GetValue<double>());
            }
            if (any_null) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
                continue;
            }
            try {
                const double r = fn(a);
                if (std::isnan(r) || std::isinf(r)) {
                    result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
                } else {
                    result.SetValue(i, duckdb::Value::DOUBLE(r));
                }
            } catch (const std::exception&) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::DOUBLE));
            }
        }
    };
    const duckdb::vector<duckdb::LogicalType> arg_types(arity, duckdb::LogicalType::DOUBLE);
    con.CreateVectorizedFunction(name, arg_types, duckdb::LogicalType::DOUBLE, udf);
}

/// Register a scalar UDF (DOUBLE x arity) -> VARCHAR (effect-size interpretation labels, ...).
inline void RegisterScalarString(duckdb::Connection& con, const std::string& name, std::size_t arity,
                                 ScalarStringFn fn) {
    duckdb::scalar_function_t udf =
        [fn, arity](duckdb::DataChunk& args, duckdb::ExpressionState& /*state*/,
                    duckdb::Vector& result) -> void {
        for (duckdb::idx_t i = 0; i < args.size(); ++i) {
            std::vector<double> a;
            a.reserve(arity);
            bool any_null = false;
            for (std::size_t j = 0; j < arity; ++j) {
                const duckdb::Value v = args.data[j].GetValue(i);
                if (v.IsNull()) {
                    any_null = true;
                    break;
                }
                a.push_back(v.GetValue<double>());
            }
            if (any_null) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::VARCHAR));
                continue;
            }
            try {
                result.SetValue(i, duckdb::Value(fn(a)));
            } catch (const std::exception&) {
                result.SetValue(i, duckdb::Value(duckdb::LogicalType::VARCHAR));
            }
        }
    };
    const duckdb::vector<duckdb::LogicalType> arg_types(arity, duckdb::LogicalType::DOUBLE);
    con.CreateVectorizedFunction(name, arg_types, duckdb::LogicalType::VARCHAR, udf);
}

// ---------------------------------------------------------------------------
// Integer-cast shorthands (cast the j-th element of the argument vector `a`)
// ---------------------------------------------------------------------------

namespace scalar_detail {
inline std::uint64_t U64(const std::vector<double>& a, std::size_t j) {
    return static_cast<std::uint64_t>(a[j]);
}
inline std::int64_t I64(const std::vector<double>& a, std::size_t j) {
    return static_cast<std::int64_t>(a[j]);
}
inline std::size_t Sz(const std::vector<double>& a, std::size_t j) {
    return static_cast<std::size_t>(a[j]);
}

/// Convert an effect-size magnitude enum to a string.
inline std::string MagnitudeToString(statcpp::effect_size_magnitude m) {
    switch (m) {
        case statcpp::effect_size_magnitude::negligible:
            return "negligible";
        case statcpp::effect_size_magnitude::small:
            return "small";
        case statcpp::effect_size_magnitude::medium:
            return "medium";
        case statcpp::effect_size_magnitude::large:
            return "large";
    }
    return "unknown";
}
}  // namespace scalar_detail

// ---------------------------------------------------------------------------
// Scalar function registration (123 functions: 31 test/helpers + 83 distributions/transforms;
// effect-size interpretation returns VARCHAR)
// ---------------------------------------------------------------------------

inline void RegisterStatcppScalarFunctions(duckdb::Connection& con) {
    using namespace scalar_detail;

    // ===== Scalar - tests / helpers (31 functions; JSON-returning ones excluded) =====

    // Normal distribution
    RegisterScalarMath(con, "stat_normal_pdf", 3,
                       [](const std::vector<double>& a) { return statcpp::normal_pdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_normal_cdf", 3,
                       [](const std::vector<double>& a) { return statcpp::normal_cdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_normal_quantile", 3, [](const std::vector<double>& a) {
        return statcpp::normal_quantile(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_normal_rand", 2,
                       [](const std::vector<double>& a) { return statcpp::normal_rand(a[0], a[1]); });

    // Chi-square distribution
    RegisterScalarMath(con, "stat_chisq_pdf", 2,
                       [](const std::vector<double>& a) { return statcpp::chisq_pdf(a[0], a[1]); });
    RegisterScalarMath(con, "stat_chisq_cdf", 2,
                       [](const std::vector<double>& a) { return statcpp::chisq_cdf(a[0], a[1]); });
    RegisterScalarMath(con, "stat_chisq_quantile", 2,
                       [](const std::vector<double>& a) { return statcpp::chisq_quantile(a[0], a[1]); });
    RegisterScalarMath(con, "stat_chisq_rand", 1,
                       [](const std::vector<double>& a) { return statcpp::chisq_rand(a[0]); });

    // t distribution
    RegisterScalarMath(con, "stat_t_pdf", 2,
                       [](const std::vector<double>& a) { return statcpp::t_pdf(a[0], a[1]); });
    RegisterScalarMath(con, "stat_t_cdf", 2,
                       [](const std::vector<double>& a) { return statcpp::t_cdf(a[0], a[1]); });
    RegisterScalarMath(con, "stat_t_quantile", 2,
                       [](const std::vector<double>& a) { return statcpp::t_quantile(a[0], a[1]); });
    RegisterScalarMath(con, "stat_t_rand", 1,
                       [](const std::vector<double>& a) { return statcpp::t_rand(a[0]); });

    // F distribution
    RegisterScalarMath(con, "stat_f_pdf", 3,
                       [](const std::vector<double>& a) { return statcpp::f_pdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_f_cdf", 3,
                       [](const std::vector<double>& a) { return statcpp::f_cdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_f_quantile", 3,
                       [](const std::vector<double>& a) { return statcpp::f_quantile(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_f_rand", 2,
                       [](const std::vector<double>& a) { return statcpp::f_rand(a[0], a[1]); });

    // Special functions (test helpers)
    RegisterScalarMath(con, "stat_betainc", 3,
                       [](const std::vector<double>& a) { return statcpp::betainc(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_betaincinv", 3,
                       [](const std::vector<double>& a) { return statcpp::betaincinv(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_norm_cdf", 1,
                       [](const std::vector<double>& a) { return statcpp::norm_cdf(a[0]); });
    RegisterScalarMath(con, "stat_norm_quantile", 1,
                       [](const std::vector<double>& a) { return statcpp::norm_quantile(a[0]); });
    RegisterScalarMath(con, "stat_gammainc_lower", 2,
                       [](const std::vector<double>& a) { return statcpp::gammainc_lower(a[0], a[1]); });
    RegisterScalarMath(con, "stat_gammainc_upper", 2,
                       [](const std::vector<double>& a) { return statcpp::gammainc_upper(a[0], a[1]); });
    RegisterScalarMath(con, "stat_gammainc_lower_inv", 2, [](const std::vector<double>& a) {
        return statcpp::gammainc_lower_inv(a[0], a[1]);
    });

    // Multiple-testing corrections (return an adjusted p-value)
    RegisterScalarMath(con, "stat_bonferroni", 2,
                       [](const std::vector<double>& a) { return std::min(a[0] * a[1], 1.0); });
    RegisterScalarMath(con, "stat_bh_correction", 3, [](const std::vector<double>& a) -> double {
        const double rank = a[1];
        if (rank <= 0.0) return NAN;
        return std::min(a[0] * a[2] / rank, 1.0);
    });
    RegisterScalarMath(con, "stat_holm_correction", 3, [](const std::vector<double>& a) {
        return std::min(a[0] * (a[2] - a[1] + 1.0), 1.0);
    });

    // 2x2 table measure (double)
    RegisterScalarMath(con, "stat_nnt", 4, [](const std::vector<double>& a) {
        const std::vector<std::vector<std::size_t>> table = {{Sz(a, 0), Sz(a, 1)},
                                                             {Sz(a, 2), Sz(a, 3)}};
        return statcpp::number_needed_to_treat(table);
    });

    // Model selection
    RegisterScalarMath(con, "stat_aic", 2,
                       [](const std::vector<double>& a) { return statcpp::aic(a[0], Sz(a, 1)); });
    RegisterScalarMath(con, "stat_aicc", 3, [](const std::vector<double>& a) {
        return statcpp::aicc(a[0], Sz(a, 1), Sz(a, 2));
    });
    RegisterScalarMath(con, "stat_bic", 3, [](const std::vector<double>& a) {
        return statcpp::bic(a[0], Sz(a, 1), Sz(a, 2));
    });

    // Box-Cox transform
    RegisterScalarMath(con, "stat_boxcox", 2, [](const std::vector<double>& a) -> double {
        const double x = a[0];
        const double lambda = a[1];
        if (std::abs(lambda) < 1e-10) return std::log(x);
        return (std::pow(x, lambda) - 1.0) / lambda;
    });

    // ===== Scalar - continuous distributions (24 functions) =====

    // Uniform distribution
    RegisterScalarMath(con, "stat_uniform_pdf", 3,
                       [](const std::vector<double>& a) { return statcpp::uniform_pdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_uniform_cdf", 3,
                       [](const std::vector<double>& a) { return statcpp::uniform_cdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_uniform_quantile", 3, [](const std::vector<double>& a) {
        return statcpp::uniform_quantile(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_uniform_rand", 2,
                       [](const std::vector<double>& a) { return statcpp::uniform_rand(a[0], a[1]); });

    // Exponential distribution
    RegisterScalarMath(con, "stat_exponential_pdf", 2,
                       [](const std::vector<double>& a) { return statcpp::exponential_pdf(a[0], a[1]); });
    RegisterScalarMath(con, "stat_exponential_cdf", 2,
                       [](const std::vector<double>& a) { return statcpp::exponential_cdf(a[0], a[1]); });
    RegisterScalarMath(con, "stat_exponential_quantile", 2, [](const std::vector<double>& a) {
        return statcpp::exponential_quantile(a[0], a[1]);
    });
    RegisterScalarMath(con, "stat_exponential_rand", 1,
                       [](const std::vector<double>& a) { return statcpp::exponential_rand(a[0]); });

    // Gamma distribution
    RegisterScalarMath(con, "stat_gamma_pdf", 3,
                       [](const std::vector<double>& a) { return statcpp::gamma_pdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_gamma_cdf", 3,
                       [](const std::vector<double>& a) { return statcpp::gamma_cdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_gamma_quantile", 3, [](const std::vector<double>& a) {
        return statcpp::gamma_quantile(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_gamma_rand", 2,
                       [](const std::vector<double>& a) { return statcpp::gamma_rand(a[0], a[1]); });

    // Beta distribution
    RegisterScalarMath(con, "stat_beta_pdf", 3,
                       [](const std::vector<double>& a) { return statcpp::beta_pdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_beta_cdf", 3,
                       [](const std::vector<double>& a) { return statcpp::beta_cdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_beta_quantile", 3, [](const std::vector<double>& a) {
        return statcpp::beta_quantile(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_beta_rand", 2,
                       [](const std::vector<double>& a) { return statcpp::beta_rand(a[0], a[1]); });

    // Log-normal distribution
    RegisterScalarMath(con, "stat_lognormal_pdf", 3, [](const std::vector<double>& a) {
        return statcpp::lognormal_pdf(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_lognormal_cdf", 3, [](const std::vector<double>& a) {
        return statcpp::lognormal_cdf(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_lognormal_quantile", 3, [](const std::vector<double>& a) {
        return statcpp::lognormal_quantile(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_lognormal_rand", 2,
                       [](const std::vector<double>& a) { return statcpp::lognormal_rand(a[0], a[1]); });

    // Weibull distribution
    RegisterScalarMath(con, "stat_weibull_pdf", 3,
                       [](const std::vector<double>& a) { return statcpp::weibull_pdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_weibull_cdf", 3,
                       [](const std::vector<double>& a) { return statcpp::weibull_cdf(a[0], a[1], a[2]); });
    RegisterScalarMath(con, "stat_weibull_quantile", 3, [](const std::vector<double>& a) {
        return statcpp::weibull_quantile(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_weibull_rand", 2,
                       [](const std::vector<double>& a) { return statcpp::weibull_rand(a[0], a[1]); });

    // ===== Scalar - discrete distributions (28 functions) =====

    // Binomial distribution
    RegisterScalarMath(con, "stat_binomial_pmf", 3, [](const std::vector<double>& a) {
        return statcpp::binomial_pmf(U64(a, 0), U64(a, 1), a[2]);
    });
    RegisterScalarMath(con, "stat_binomial_cdf", 3, [](const std::vector<double>& a) {
        return statcpp::binomial_cdf(U64(a, 0), U64(a, 1), a[2]);
    });
    RegisterScalarMath(con, "stat_binomial_quantile", 3, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::binomial_quantile(a[0], U64(a, 1), a[2]));
    });
    RegisterScalarMath(con, "stat_binomial_rand", 2, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::binomial_rand(U64(a, 0), a[1]));
    });

    // Poisson distribution
    RegisterScalarMath(con, "stat_poisson_pmf", 2, [](const std::vector<double>& a) {
        return statcpp::poisson_pmf(U64(a, 0), a[1]);
    });
    RegisterScalarMath(con, "stat_poisson_cdf", 2, [](const std::vector<double>& a) {
        return statcpp::poisson_cdf(U64(a, 0), a[1]);
    });
    RegisterScalarMath(con, "stat_poisson_quantile", 2, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::poisson_quantile(a[0], a[1]));
    });
    RegisterScalarMath(con, "stat_poisson_rand", 1, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::poisson_rand(a[0]));
    });

    // Geometric distribution
    RegisterScalarMath(con, "stat_geometric_pmf", 2, [](const std::vector<double>& a) {
        return statcpp::geometric_pmf(U64(a, 0), a[1]);
    });
    RegisterScalarMath(con, "stat_geometric_cdf", 2, [](const std::vector<double>& a) {
        return statcpp::geometric_cdf(U64(a, 0), a[1]);
    });
    RegisterScalarMath(con, "stat_geometric_quantile", 2, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::geometric_quantile(a[0], a[1]));
    });
    RegisterScalarMath(con, "stat_geometric_rand", 1, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::geometric_rand(a[0]));
    });

    // Negative binomial distribution
    RegisterScalarMath(con, "stat_nbinom_pmf", 3, [](const std::vector<double>& a) {
        return statcpp::nbinom_pmf(U64(a, 0), a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_nbinom_cdf", 3, [](const std::vector<double>& a) {
        return statcpp::nbinom_cdf(U64(a, 0), a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_nbinom_quantile", 3, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::nbinom_quantile(a[0], a[1], a[2]));
    });
    RegisterScalarMath(con, "stat_nbinom_rand", 2, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::nbinom_rand(a[0], a[1]));
    });

    // Hypergeometric distribution
    RegisterScalarMath(con, "stat_hypergeom_pmf", 4, [](const std::vector<double>& a) {
        return statcpp::hypergeom_pmf(U64(a, 0), U64(a, 1), U64(a, 2), U64(a, 3));
    });
    RegisterScalarMath(con, "stat_hypergeom_cdf", 4, [](const std::vector<double>& a) {
        return statcpp::hypergeom_cdf(U64(a, 0), U64(a, 1), U64(a, 2), U64(a, 3));
    });
    RegisterScalarMath(con, "stat_hypergeom_quantile", 4, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::hypergeom_quantile(a[0], U64(a, 1), U64(a, 2), U64(a, 3)));
    });
    RegisterScalarMath(con, "stat_hypergeom_rand", 3, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::hypergeom_rand(U64(a, 0), U64(a, 1), U64(a, 2)));
    });

    // Bernoulli distribution
    RegisterScalarMath(con, "stat_bernoulli_pmf", 2, [](const std::vector<double>& a) {
        return statcpp::bernoulli_pmf(U64(a, 0), a[1]);
    });
    RegisterScalarMath(con, "stat_bernoulli_cdf", 2, [](const std::vector<double>& a) {
        return statcpp::bernoulli_cdf(U64(a, 0), a[1]);
    });
    RegisterScalarMath(con, "stat_bernoulli_quantile", 2, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::bernoulli_quantile(a[0], a[1]));
    });
    RegisterScalarMath(con, "stat_bernoulli_rand", 1, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::bernoulli_rand(a[0]));
    });

    // Discrete uniform distribution
    RegisterScalarMath(con, "stat_duniform_pmf", 3, [](const std::vector<double>& a) {
        return statcpp::discrete_uniform_pmf(I64(a, 0), I64(a, 1), I64(a, 2));
    });
    RegisterScalarMath(con, "stat_duniform_cdf", 3, [](const std::vector<double>& a) {
        return statcpp::discrete_uniform_cdf(I64(a, 0), I64(a, 1), I64(a, 2));
    });
    RegisterScalarMath(con, "stat_duniform_quantile", 3, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::discrete_uniform_quantile(a[0], I64(a, 1), I64(a, 2)));
    });
    RegisterScalarMath(con, "stat_duniform_rand", 2, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::discrete_uniform_rand(I64(a, 0), I64(a, 1)));
    });

    // ===== Scalar - combinatorics / special functions (9 functions) =====

    RegisterScalarMath(con, "stat_binomial_coef", 2, [](const std::vector<double>& a) {
        return statcpp::binomial_coef(U64(a, 0), U64(a, 1));
    });
    RegisterScalarMath(con, "stat_log_binomial_coef", 2, [](const std::vector<double>& a) {
        return statcpp::log_binomial_coef(U64(a, 0), U64(a, 1));
    });
    RegisterScalarMath(con, "stat_log_factorial", 1,
                       [](const std::vector<double>& a) { return statcpp::log_factorial(U64(a, 0)); });
    RegisterScalarMath(con, "stat_lgamma", 1,
                       [](const std::vector<double>& a) { return statcpp::lgamma(a[0]); });
    RegisterScalarMath(con, "stat_tgamma", 1,
                       [](const std::vector<double>& a) { return statcpp::tgamma(a[0]); });
    RegisterScalarMath(con, "stat_beta_func", 2,
                       [](const std::vector<double>& a) { return statcpp::beta(a[0], a[1]); });
    RegisterScalarMath(con, "stat_lbeta", 2,
                       [](const std::vector<double>& a) { return statcpp::lbeta(a[0], a[1]); });
    RegisterScalarMath(con, "stat_erf", 1,
                       [](const std::vector<double>& a) { return statcpp::erf(a[0]); });
    RegisterScalarMath(con, "stat_erfc", 1,
                       [](const std::vector<double>& a) { return statcpp::erfc(a[0]); });

    // ===== Scalar - basic statistics (1 function) =====

    RegisterScalarMath(con, "stat_logarithmic_mean", 2, [](const std::vector<double>& a) -> double {
        const double x = a[0];
        const double y = a[1];
        if (x <= 0.0 || y <= 0.0) return NAN;
        if (std::abs(x - y) < 1e-15) return x;
        return (y - x) / (std::log(y) - std::log(x));
    });

    // ===== Scalar - effect-size corrections / conversions (8 functions) =====

    RegisterScalarMath(con, "stat_hedges_j", 1, [](const std::vector<double>& a) {
        return statcpp::hedges_correction_factor(a[0]);
    });
    RegisterScalarMath(con, "stat_t_to_r", 2,
                       [](const std::vector<double>& a) { return statcpp::t_to_r(a[0], a[1]); });
    RegisterScalarMath(con, "stat_d_to_r", 1,
                       [](const std::vector<double>& a) { return statcpp::d_to_r(a[0]); });
    RegisterScalarMath(con, "stat_r_to_d", 1,
                       [](const std::vector<double>& a) { return statcpp::r_to_d(a[0]); });
    RegisterScalarMath(con, "stat_eta_squared_ef", 2,
                       [](const std::vector<double>& a) { return statcpp::eta_squared(a[0], a[1]); });
    RegisterScalarMath(con, "stat_partial_eta_sq", 3, [](const std::vector<double>& a) {
        return statcpp::partial_eta_squared(a[0], a[1], a[2]);
    });
    RegisterScalarMath(con, "stat_omega_squared_ef", 4, [](const std::vector<double>& a) {
        return statcpp::omega_squared(a[0], a[1], a[2], a[3]);
    });
    RegisterScalarMath(con, "stat_cohens_h", 2,
                       [](const std::vector<double>& a) { return statcpp::cohens_h(a[0], a[1]); });

    // ===== Scalar - effect-size interpretation (3 functions, VARCHAR) =====

    RegisterScalarString(con, "stat_interpret_d", 1, [](const std::vector<double>& a) {
        return MagnitudeToString(statcpp::interpret_cohens_d(a[0]));
    });
    RegisterScalarString(con, "stat_interpret_r", 1, [](const std::vector<double>& a) {
        return MagnitudeToString(statcpp::interpret_correlation(a[0]));
    });
    RegisterScalarString(con, "stat_interpret_eta2", 1, [](const std::vector<double>& a) {
        return MagnitudeToString(statcpp::interpret_eta_squared(a[0]));
    });

    // ===== Scalar - power analysis (6 functions) =====

    RegisterScalarMath(con, "stat_power_t1", 3, [](const std::vector<double>& a) {
        return statcpp::power_t_test_one_sample(a[0], Sz(a, 1), a[2]);
    });
    RegisterScalarMath(con, "stat_n_t1", 3, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::sample_size_t_test_one_sample(a[0], a[1], a[2]));
    });
    RegisterScalarMath(con, "stat_power_t2", 4, [](const std::vector<double>& a) {
        return statcpp::power_t_test_two_sample(a[0], Sz(a, 1), Sz(a, 2), a[3]);
    });
    RegisterScalarMath(con, "stat_n_t2", 3, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::sample_size_t_test_two_sample(a[0], a[1], a[2]));
    });
    RegisterScalarMath(con, "stat_power_prop", 4, [](const std::vector<double>& a) {
        return statcpp::power_prop_test(a[0], a[1], Sz(a, 2), a[3]);
    });
    RegisterScalarMath(con, "stat_n_prop", 4, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::sample_size_prop_test(a[0], a[1], a[2], a[3]));
    });

    // ===== Scalar - margin of error / sample size (4 functions) =====

    RegisterScalarMath(con, "stat_moe_prop", 3, [](const std::vector<double>& a) {
        return statcpp::margin_of_error_proportion(Sz(a, 0), Sz(a, 1), a[2]);
    });
    RegisterScalarMath(con, "stat_moe_prop_worst", 2, [](const std::vector<double>& a) {
        return statcpp::margin_of_error_proportion_worst_case(Sz(a, 0), a[1]);
    });
    RegisterScalarMath(con, "stat_n_moe_prop", 3, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::sample_size_for_moe_proportion(a[0], a[1], a[2]));
    });
    RegisterScalarMath(con, "stat_n_moe_mean", 3, [](const std::vector<double>& a) {
        return static_cast<double>(statcpp::sample_size_for_moe_mean(a[0], a[1], a[2]));
    });
}

// ---------------------------------------------------------------------------
// Overall umbrella
// ---------------------------------------------------------------------------

/**
 * @brief Register all statcpp UDFs (LIST-based + scalar).
 *
 * Statistics are typically applied to a LIST built with list(), e.g.
 *
 *     SELECT stat_median(list(v)) FROM t;
 *     SELECT stat_percentile(list(v), 0.9) FROM t;
 *     SELECT stat_pearson_r(list(x), list(y)) FROM t;
 *     SELECT unnest(stat_rolling_mean(list(v ORDER BY id), 3)) FROM t;
 *     SELECT stat_normal_quantile(0.975, 0, 1);          -- scalar
 */
inline void RegisterStatcppFunctions(duckdb::Connection& con) {
    RegisterStatcppListFunctions(con);
    RegisterStatcppScalarFunctions(con);
}

}  // namespace statcpp_duckdb
