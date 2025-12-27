# %%
import pandas as pd
from pathlib import Path
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis, norm
from scipy.stats import pearsonr
import statsmodels.api as sm
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from linearmodels.panel import RandomEffects
from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# %%
final_data_path = Path("/Users/lorenzouberti/Desktop/Panel-Data-Project-Financial-Econometrics/Data")
df_Final = pd.read_excel(final_data_path / "final_panel_dataset.xlsx")

df_Final = df_Final.sort_values(["country", "year"])
df_Final = df_Final[['country', 'year', 'gdppc', 'labour_productivity', 'cpi', 'trade_openness', 'liquid_liabilities', 'mcap_gdp', 'bank_loans_gdp']]
df_Final = df_Final.sort_values(["country", "year"])

# Logging GDP per capita first
df_Final['ln_gdppc'] = np.log(df_Final['gdppc'])
df_Final["ln_gdppc_lag_2"] = df_Final.groupby("country")["ln_gdppc"].shift(2)
# Creating the 5 year lagged GDP per capita growth variable
df_Final["ln_gdp_growth_2y"] = df_Final["ln_gdppc"] - df_Final["ln_gdppc_lag_2"]

# Doing of rthe alternative dependent variable: labour productivity growth
df_Final['ln_labour_productivity'] = np.log(df_Final['labour_productivity'])
df_Final["ln_labour_productivity_lag2"] = df_Final.groupby("country")["ln_labour_productivity"].shift(2)
df_Final["ln_labour_productivity_growth_2y"] = df_Final["ln_labour_productivity"] - df_Final["ln_labour_productivity_lag2"]

# taking teh logarithm of the other variables as well
df_Final['ln_cpi'] = np.log(df_Final['cpi'])
df_Final['ln_trade_openness'] = np.log(df_Final['trade_openness'])

variables = [ 'ln_gdppc','ln_labour_productivity', 'ln_cpi', 'ln_trade_openness', 'liquid_liabilities', 'mcap_gdp', 'bank_loans_gdp', "ln_gdp_growth_2y", "ln_labour_productivity_growth_2y" ]

y = 'ln_gdp_growth_2y'  
X = ['liquid_liabilities', 'mcap_gdp', 'bank_loans_gdp']
Con = ['ln_cpi', 'ln_trade_openness']

# Splitting my data into the seperate regimes -> pre 1928, 1929–1971, post 1971

df_Regime = df_Final.copy()

conditions = [
	(df_Regime["year"] < 1928),
	(df_Regime["year"] >= 1928) & (df_Regime["year"] <= 1971),
	(df_Regime["year"] >= 1974)
]
regimes = ['Gold Standard', 'Bretton Woods','Finanzialisation']

regime_arr = np.select(conditions, regimes, default='')
df_Regime['regime'] = pd.Series(regime_arr, index=df_Regime.index).replace('', np.nan)

df_GS = df_Regime[df_Regime['regime'] == 'Gold Standard']
df_BW = df_Regime[df_Regime['regime'] == 'Bretton Woods']
df_Fin = df_Regime[df_Regime['regime'] == 'Finanzialisation']







# %%
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

def poolability_test(df, y, x, controls, entity):

    rhs_vars = x + controls
    rhs_common = " + ".join(rhs_vars)

    # Restricted
    restricted_formula = f"{y} ~ {rhs_common} + C({entity})"

    # Unrestricted
    interactions = " + ".join([f"{var}:C({entity})" for var in rhs_vars])
    unrestricted_formula = (
        f"{y} ~ {rhs_common} + C({entity}) + {interactions}"
    )

    #  estimate models
    m_r = smf.ols(restricted_formula, data=df).fit()
    m_u = smf.ols(unrestricted_formula, data=df).fit()

    # SSR
    SSR_r = np.sum(m_r.resid ** 2)
    SSR_u = np.sum(m_u.resid ** 2)

    # dof
    q = int(m_u.df_model - m_r.df_model)   # number of restrictions
    df_u = int(m_u.df_resid)

    # F star
    F = ((SSR_r - SSR_u) / q) / (SSR_u / df_u)
    p_value = 1 - stats.f.cdf(F, q, df_u)

    return {
        "Restricted formula": restricted_formula,
        "Unrestricted formula": unrestricted_formula,
        "F statistic": F,
        "p-value": p_value,
        "q restrictions": q,
        "df residual (unrestricted)": df_u
    }

y = "ln_gdp_growth_2y"
X = ['liquid_liabilities', 'mcap_gdp', 'bank_loans_gdp']
Con = ['ln_cpi', 'ln_trade_openness']

results = {}

for name, df_sub in {
    "Gold Standard (GS)": df_GS,
    "Bretton Woods (BW)": df_BW,
    "Financialization (FIN)": df_Fin
}.items():

    res = poolability_test(
        df=df_sub,
        y=y,
        x=X,
        controls=Con,
        entity="country"
    )

    results[name] = res

    print(f"\n{name}")
    print("-" * 50)
    print(f"F-statistic : {res['F statistic']:.3f}")
    print(f"p-value     : {res['p-value']:.4f}")
    print(f"Restrictions: {res['q restrictions']}")




# %%
# Adf Test for stationarity

def adf_test(series, maxlag=None, regression = 'c'):
    
    series = series.dropna()
    if len(series) < 10:
        return None
    result = adfuller(series, maxlag=maxlag, regression=regression, autolag="AIC")
    return {
        "ADF statistic": result[0],
        "p-value": result[1],
        "used lags": result[2],
        "nobs": result[3],
        "5% crit": result[4]["5%"],
        "stationary_5pct": result[1] < 0.05
    }

vars_to_test = [
    "ln_gdp_growth_2y",
    "liquid_liabilities",
    "bank_loans_gdp",
    "mcap_gdp",
    "ln_cpi",
    "ln_trade_openness"
]

def run_adf_by_country(df, variables, regime_name):
    results = []

    for country in df["country"].unique():
        df_c = df[df["country"] == country]

        for var in variables:
            res = adf_test(df_c[var], regression="c")
            if res is None:
                continue

            results.append({
                "regime": regime_name,
                "country": country,
                "variable": var,
                "ADF stat": res["ADF statistic"],
                "p-value": res["p-value"],
                "stationary (5%)": res["stationary_5pct"],
                "nobs": res["nobs"]
            })

    return pd.DataFrame(results)

adf_GS  = run_adf_by_country(df_GS,  vars_to_test, "GS")
adf_BW  = run_adf_by_country(df_BW,  vars_to_test, "BW")
adf_FIN = run_adf_by_country(df_Fin, vars_to_test, "FIN")

adf_results = pd.concat([adf_GS, adf_BW, adf_FIN], ignore_index=True)

adf_results.head()


# %%
stationarity_by_var = (
    adf_results
    .groupby("variable")["stationary (5%)"]
    .mean()
    .reset_index()
    .rename(columns={"stationary (5%)": "share_stationary"})
)

stationarity_by_var["share_nonstationary"] = 1 - stationarity_by_var["share_stationary"]

print(stationarity_by_var)

# %%
#serial Autocorrelation test 

def bg_test_one_country(df_country, y, x, controls, nlags=2):
    rhs = " + ".join(x + controls)
    formula = f"{y} ~ {rhs}"

    
    model = smf.ols(formula, data=df_country).fit()

    # Breusch–Godfrey test
    lm_stat, lm_pvalue, f_stat, f_pvalue = acorr_breusch_godfrey(model, nlags=nlags)

    return {
        "LM stat": lm_stat,
        "LM p-value": lm_pvalue,
        "F stat": f_stat,
        "F p-value": f_pvalue,
        "nobs": int(model.nobs)
    }

def bg_test_by_regime(df_regime, regime_name, y, x, controls, nlags=2, min_obs=15):
    results = []

    for country in sorted(df_regime["country"].unique()):
        df_c = df_regime[df_regime["country"] == country].dropna(subset=[y] + x + controls)

        # skip if too few observations
        if len(df_c) < min_obs:
            continue

        out = bg_test_one_country(df_c, y, x, controls, nlags=nlags)
        results.append({
            "regime": regime_name,
            "country": country,
            "nlags": nlags,
            **out
        })

    return pd.DataFrame(results)

y = "ln_gdp_growth_2y"
X = ['liquid_liabilities', 'mcap_gdp', 'bank_loans_gdp']
Con = ['ln_cpi', 'ln_trade_openness']

# Choose lag order for BG test (2 is a common starting point; try 1–4 as robustness)
nlags = 2

bg_GS  = bg_test_by_regime(df_GS,  "GS",  y, X, Con, nlags=nlags)
bg_BW  = bg_test_by_regime(df_BW,  "BW",  y, X, Con, nlags=nlags)
bg_FIN = bg_test_by_regime(df_Fin, "FIN", y, X, Con, nlags=nlags)

bg_results = pd.concat([bg_GS, bg_BW, bg_FIN], ignore_index=True)

# Flag rejection of no autocorrelation
bg_results["reject_no_autocorr_5pct"] = bg_results["LM p-value"] < 0.05

# Summary: share of countries rejecting no-autocorrelation by regime
summary_bg = (
    bg_results.groupby("regime")["reject_no_autocorr_5pct"]
    .mean()
    .reset_index(name="share_reject_BG_5pct")
)

print(summary_bg)

# Optional: view the country-level results (sorted by p-value)
print(bg_results.sort_values(["regime", "LM p-value"]).head())
print(bg_results)

# %%
# testing to see if there are some structural breask int tyeh data set 

formula = f"{y} ~ " + " + ".join(X + Con)


def chow_test_known_break(df_country, formula, break_year):

    df_country = df_country.dropna().copy()
    df_country = df_country.sort_values("year")

    df_pre  = df_country[df_country["year"] <= break_year]
    df_post = df_country[df_country["year"] > break_year]

    # Need enough obs on both sides
    if len(df_pre) < 10 or len(df_post) < 10:
        return None

    m_full = smf.ols(formula, data=df_country).fit()
    m_pre  = smf.ols(formula, data=df_pre).fit()
    m_post = smf.ols(formula, data=df_post).fit()

    SSR_full = float(np.sum(m_full.resid**2))
    SSR_pre  = float(np.sum(m_pre.resid**2))
    SSR_post = float(np.sum(m_post.resid**2))

    k = int(m_full.df_model + 1)  # number of parameters incl intercept
    n = int(m_full.nobs)

    # Chow F-stat
    num = (SSR_full - (SSR_pre + SSR_post)) / k
    den = (SSR_pre + SSR_post) / (n - 2*k)
    if den <= 0:
        return None

    F = num / den
    p = 1 - stats.f.cdf(F, k, n - 2*k)

    return {"break_year": break_year, "F": F, "p_value": p, "n": n, "k": k}

breaks_by_regime = {
    "GS":  [1893, 1907, 1914],
    "BW":  [1933, 1945, 1958],
    "FIN": [1987, 2001, 2008],
}

def chow_tests_regime(df_regime, regime_name, formula, break_years):
    out = []
    for country in sorted(df_regime["country"].unique()):
        df_c = df_regime[df_regime["country"] == country].copy()
        for by in break_years:
            res = chow_test_known_break(df_c, formula, by)
            if res is None:
                continue
            out.append({"regime": regime_name, "country": country, **res})
    return pd.DataFrame(out)

chow_GS  = chow_tests_regime(df_GS,  "GS",  formula, breaks_by_regime["GS"])
chow_BW  = chow_tests_regime(df_BW,  "BW",  formula, breaks_by_regime["BW"])
chow_FIN = chow_tests_regime(df_Fin, "FIN", formula, breaks_by_regime["FIN"])

chow_results = pd.concat([chow_GS, chow_BW, chow_FIN], ignore_index=True)

# Flag significant breaks
chow_results["reject_stability_5pct"] = chow_results["p_value"] < 0.05

# Summary table: share of (country × breakdate) tests rejecting stability
chow_summary = (
    chow_results.groupby("regime")["reject_stability_5pct"]
    .mean().reset_index(name="share_reject_chow_5pct")
)
print(chow_summary)

# (Optional) see the strongest breaks
print(chow_results.sort_values("p_value").head(15))


# %%
y = "ln_gdp_growth_2y"

finance_vars = ["liquid_liabilities", "bank_loans_gdp", "mcap_gdp"]
controls = ["ln_cpi", "ln_trade_openness"]
base_rhs = finance_vars + controls

def add_break_dummies(df):
    df = df.copy()
    df["D_1914"] = (df["year"] >= 1914).astype(int)
    df["D_1945"] = (df["year"] >= 1945).astype(int)
    df["D_1987"] = (df["year"] >= 1987).astype(int)
    return df

df_GS  = add_break_dummies(df_GS)
df_BW  = add_break_dummies(df_BW)
df_FIN = add_break_dummies(df_Fin)   

def country_hac_regressions(df, regime, breaks, maxlags=2, min_obs=15):
    rows = []
    rhs = base_rhs + breaks
    formula = f"{y} ~ " + " + ".join(rhs)

    # columns required for estimation (so we can dropna explicitly)
    required = [y] + rhs + ["country", "year"]

    for c in sorted(df["country"].unique()):
        df_c = df[df["country"] == c].copy()

        # Drop missing *before* calling statsmodels
        df_c = df_c.dropna(subset=required)

        # Need enough obs after dropping missing
        if len(df_c) < min_obs:
            continue

        # HAC maxlags must be < number of observations (rule of thumb)
        ml = min(maxlags, max(1, len(df_c)//4))

        try:
            model = smf.ols(formula, data=df_c).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": ml}
            )
        except Exception as e:
            # Keep track of failures rather than crashing
            rows.append({
                "regime": regime,
                "country": c,
                "variable": "__MODEL_FAILED__",
                "coef": np.nan,
                "se": np.nan,
                "t": np.nan,
                "p": np.nan,
                "error": str(e),
                "nobs": len(df_c)
            })
            continue

        for v in finance_vars:
            rows.append({
                "regime": regime,
                "country": c,
                "variable": v,
                "coef": model.params.get(v, np.nan),
                "se": model.bse.get(v, np.nan),
                "t": model.tvalues.get(v, np.nan),
                "p": model.pvalues.get(v, np.nan),
                "nobs": int(model.nobs),
                "maxlags_used": ml
            })

    return pd.DataFrame(rows)

res_GS  = country_hac_regressions(df_GS,  "GS",  breaks=["D_1914"])
res_BW  = country_hac_regressions(df_BW,  "BW",  breaks=["D_1945"])
res_FIN = country_hac_regressions(df_Fin, "FIN", breaks=["D_1987"])

results = pd.concat([res_GS, res_BW, res_FIN], ignore_index=True)

# Optional: see if any country models failed
failed = results[results["variable"] == "__MODEL_FAILED__"]
print("Model failures:", len(failed))
if len(failed) > 0:
    print(failed[["regime","country","nobs","error"]].head(10))

# Keep only the finance coefficient rows for hypothesis testing
results_finance = results[results["variable"].isin(finance_vars)].copy()
print(results_finance)

# %%
H1_test = (
    results
    .groupby(["regime", "variable"])["coef"]
    .mean()
    .unstack("regime")
)

print("H1 – Mean coefficients by regime")
print(H1_test)

# %%
H2_test = (
    results[results["variable"].isin(finance_vars)]
    .groupby(["regime", "variable"])["coef"]
    .mean()
    .reset_index()
)

print("H2 – Average finance coefficients by regime")
print(H2_test)

# %%
H3_test = (
    results[(results["regime"] == "BW") &
            (results["variable"].isin(["bank_loans_gdp", "mcap_gdp"]))]
    .groupby("variable")["coef"]
    .agg(["mean", "std", "count"])
)

print("H3 – Bretton Woods: credit vs markets")
print(H3_test)

# %%
H4_test = (
    results[(results["regime"] == "FIN") &
            (results["variable"] == "mcap_gdp")]
)

print("H4 – Financialization: market finance")
print(H4_test[["country", "coef", "p"]])

print("Share negative (FIN, mcap):",
      (H4_test["coef"] < 0).mean())

# %%
def engle_granger_by_country(df, y_level, x_level, entity="country", trend="c", min_obs=25):
    rows = []
    for c in sorted(df[entity].unique()):
        d = df[df[entity] == c].dropna(subset=[y_level, x_level]).sort_values("year")
        if len(d) < min_obs:
            continue

        y = d[y_level].astype(float).to_numpy()
        x = d[x_level].astype(float).to_numpy()

        # coint returns: t_stat, p_value, critical_values
        t_stat, pval, crit = coint(y, x, trend=trend)
        rows.append({
            "country": c,
            "y": y_level,
            "x": x_level,
            "nobs": len(d),
            "EG_tstat": t_stat,
            "p_value": pval,
            "coint_5pct": pval < 0.05
        })

    return pd.DataFrame(rows)


pairs = ["mcap_gdp", "bank_loans_gdp", "liquid_liabilities"]

eg_results = []
for xvar in pairs:
    out = engle_granger_by_country(df_Final, y_level="ln_gdppc", x_level=xvar, trend="c", min_obs=25)
    eg_results.append(out)

eg_results = pd.concat(eg_results, ignore_index=True)

# Summary: share of countries with cointegration (5%)
summary_eg = eg_results.groupby("x")["coint_5pct"].mean().reset_index(name="share_cointegrated_5pct")
print(eg_results)

# Optional: see country-level results
print(eg_results.sort_values(["x","p_value"]).head(20))



