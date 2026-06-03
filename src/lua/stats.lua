-- stats.lua
-- Statistical functions implemented in Lua using the statcpp C binding.
--
-- This script is loaded at runtime by lua_udf.hpp (RegisterLuaStatcppFunctions).
-- Each function defined here becomes a DuckDB UDF named "lua_stat_<name>".
--
-- Extending the system means editing this file and re-running the binary —
-- no C++ recompilation is required.

local statcpp = require("statcpp")

--- Median Absolute Deviation (robust scale estimator, unscaled).
-- @param data table  1-indexed array of numbers; nil entries = missing values
-- @return number | nil
function lua_mad(data)
    return statcpp.mad(data)
end

--- Hodges-Lehmann robust location estimator.
-- @param data table  1-indexed array of numbers; nil entries = missing values
-- @return number | nil
function lua_hodges_lehmann(data)
    return statcpp.hodges_lehmann(data)
end

--- Fraction of missing (nil) values in the input.
-- @param data table  1-indexed array of numbers; nil entries = missing values
-- @return number | nil  (value in [0, 1])
function lua_missing_rate(data)
    return statcpp.missing_rate(data)
end

--- Replace each missing value with the mean of the observed values.
-- @param data table  1-indexed array of numbers; nil entries = missing values
-- @return table  new array of the same length with missing values filled
function lua_impute_mean(data)
    return statcpp.impute_mean(data)
end

-- ---------------------------------------------------------------------------
-- Demonstrations of Lua's value: composing primitives and encoding policy.
-- These are NEW functions that do not exist in statcpp; they are defined here
-- purely in Lua, on top of the bound primitives, with NO C++ recompilation.
-- ---------------------------------------------------------------------------

--- Robust coefficient of variation: a derived metric composed in Lua.
-- Combines two statcpp primitives plus arithmetic into a single new statistic:
--   robust_cv = mad_scaled / hodges_lehmann = (1.4826 * mad) / HL
-- This metric is not provided by statcpp as one call; it is assembled here.
-- @param data table  1-indexed array of numbers; nil entries = missing values
-- @return number | nil  (nil if inputs are unavailable or the location is 0)
function lua_robust_cv(data)
    local scale = statcpp.mad(data)
    local loc = statcpp.hodges_lehmann(data)
    if scale == nil or loc == nil or loc == 0 then
        return nil
    end
    return (1.4826 * scale) / loc
end

--- Smart imputation: a policy/business rule expressed in Lua.
-- Branches on the missing rate to decide what to do — the kind of decision
-- logic that would otherwise require recompiling C++:
--   rate > 0.5  -> refuse (too unreliable), return nil
--   rate == 0   -> nothing to impute, return the data unchanged
--   otherwise   -> fill missing values with the observed mean
-- The threshold below is editable here and takes effect on the next run, with
-- no rebuild.
-- @param data table  1-indexed array of numbers; nil entries = missing values
-- @return table | nil  imputed array, or nil if imputation is refused
local SMART_IMPUTE_MAX_MISSING_RATE = 0.5

function lua_smart_impute(data)
    local rate = statcpp.missing_rate(data)
    if rate == nil or rate > SMART_IMPUTE_MAX_MISSING_RATE then
        return nil
    end
    if rate == 0 then
        return data
    end
    return statcpp.impute_mean(data)
end

--- Free-form summary report of representative values, formatted in Lua.
-- Demonstrates that Lua can freely edit/assemble the output: it mixes the usual
-- representative statistics (mean, median, min, max, range, plus the robust MAD
-- from statcpp) with values that are NOT part of a standard summary at all
-- (a skew direction label, a data-quality grade, the missing count). The set of
-- fields, their order and the wording are entirely up to this script.
-- A missing value arrives as NaN (v ~= v); it is counted, not used in the math.
-- @param data table  1-indexed array of numbers; NaN entries = missing values
-- @return string  a human-readable one-line report
function lua_summary(data)
    -- Split observed values from missing (NaN) ones.
    local xs = {}
    local missing = 0
    for _, v in ipairs(data) do
        if v ~= v then              -- NaN means missing
            missing = missing + 1
        else
            xs[#xs + 1] = v
        end
    end

    local n = #xs
    if n == 0 then
        return string.format("n=0 missing=%d | (no observed values)", missing)
    end

    -- Standard representative values (computed in pure Lua).
    table.sort(xs)
    local sum = 0.0
    for _, v in ipairs(xs) do sum = sum + v end
    local mean = sum / n
    local mn, mx = xs[1], xs[n]
    local median
    if n % 2 == 1 then
        median = xs[(n + 1) // 2]
    else
        median = (xs[n // 2] + xs[n // 2 + 1]) / 2.0
    end
    local range = mx - mn

    -- A robust value sourced from statcpp (composition with the C++ library).
    local mad = statcpp.mad(data) or (0 / 0)

    -- Freely chosen extras that are NOT standard summary statistics:
    local skew = "symmetric"
    if mean > median then skew = "right-skewed"
    elseif mean < median then skew = "left-skewed" end
    local grade = "A"
    if missing > n * 0.2 then grade = "C"
    elseif missing > 0 then grade = "B" end

    return string.format(
        "n=%d missing=%d | mean=%.2f median=%.2f min=%.2f max=%.2f range=%.2f mad=%.2f"
        .. " | shape=%s quality=%s",
        n, missing, mean, median, mn, mx, range, mad, skew, grade)
end
