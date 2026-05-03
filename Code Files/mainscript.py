##############################################################################
##############################################################################
############# Replication Code: "Coming of Age Under Trump" #################
######################### ANES RESTRICTED DATA ##############################
##############################################################################
##############################################################################

# Author: Nikolaos Vichos
# Python translation of usa_thesis_restricted.R
# Covers: H1, H2, H3, index construction, and robustness checks
# Excludes: H4 (in-party/out-party win expectations), Monte Carlo simulations

##############################################################################
# SECTION 1: SETUP
##############################################################################

import os
import warnings
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
from factor_analyzer import FactorAnalyzer
from rdrobust import rdrobust

warnings.filterwarnings("ignore")

# File paths
loc = "/Users/nikolaosvichos/Library/Mobile Documents/com~apple~CloudDocs/Sciences Po/Thesis/"
results_path = os.path.join(loc, "Thesis-Github/Thesis-Coding/Results/")

##############################################################################
# SECTION 2: DATA IMPORT AND CLEANING
##############################################################################

# --- Import ---
df_unclean = pd.read_stata(os.path.join(loc, "Datasets/ANES/anes_data.dta"))

# --- Variable selection ---
rename_map = {
    "V200003": "panel_data",
    "V200004": "pre_or_post",
    "V201507x": "age",
    "V201231x": "party",
    "V201511x": "education_summary",
    "V201510": "education",
    "V201600": "sex",
    "V202468x": "income",
    "V201549x": "race",
    "V201200": "lib_or_con",
    "V201225x": "duty_or_choice",
    "V201104": "voted2012",
    "V201101": "voted2016",
    "V201105": "voted_for_2012",
    "V201103": "voted_for_2016",
    "V201217": "expectations_2020",
    "V201156": "feeling_dems",
    "V201157": "feeling_reps",
    "V201366": "free_press",
    "V201367": "checks_and_balances",
    "V201368": "rule_of_law",
    "V201369": "agree_on_facts",
    "V201372x": "unitary_executive",
    "V201375x": "journalist_access",
    "V201376": "media_undermined_concern",
    "V201377": "media_trust",
    "V201233": "govt_trust",
    "V201234": "govt_capture",
    "V201236": "govt_corruption",
    "V201378": "foreign_help",
    "V201379": "govt_principled",
    "V201429": "urban_unrest",
}

df_unstandardized = df_unclean[list(rename_map.keys())].rename(columns=rename_map).copy()

# --- Filter to panel respondents, pre- or post-survey only ---
df_unstandardized = df_unstandardized[df_unstandardized["panel_data"].isin([3, 4, 5, 6])]
df_unstandardized = df_unstandardized[df_unstandardized["pre_or_post"].isin([1, 3])]

# --- Demographic variables ---
df_unstandardized["age"] = df_unstandardized["age"].where(df_unstandardized["age"] != -9, np.nan)

df_unstandardized["party"] = df_unstandardized["party"].where(
    ~df_unstandardized["party"].isin([-9, -8, 3]), np.nan
)


def make_party_summary(p):
    if pd.isna(p):
        return np.nan
    p = int(p)
    if p in [1, 2]:
        return "Democrat"
    elif p in [3, 4, 5]:
        return "Independent"
    elif p in [6, 7]:
        return "Republican"
    return np.nan


df_unstandardized["party_summary"] = df_unstandardized["party"].apply(make_party_summary)

df_unstandardized["education_summary"] = df_unstandardized["education_summary"].where(
    ~df_unstandardized["education_summary"].isin([-9, -8, -2]), np.nan
)
df_unstandardized["education"] = df_unstandardized["education"].where(
    ~df_unstandardized["education"].isin([-9, -8, 95]), np.nan
)

df_unstandardized["sex"] = df_unstandardized["sex"].where(
    df_unstandardized["sex"] != -9, np.nan
)
df_unstandardized["sex"] = df_unstandardized["sex"].replace(2, 0)  # 2 -> 0 (female)

df_unstandardized["income"] = df_unstandardized["income"].where(
    ~df_unstandardized["income"].isin([-9, -5]), np.nan
)

df_unstandardized["race"] = df_unstandardized["race"].where(
    ~df_unstandardized["race"].isin([-9, -8]), np.nan
)
race_map = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Native", 6: "Other"}
df_unstandardized["race"] = df_unstandardized["race"].map(race_map)

# --- Vote-related variables ---
df_unstandardized["duty_or_choice"] = df_unstandardized["duty_or_choice"].where(
    df_unstandardized["duty_or_choice"] != -2, np.nan
)

df_unstandardized["voted2012"] = df_unstandardized["voted2012"].where(
    ~df_unstandardized["voted2012"].isin([-9, -8]), np.nan
)
df_unstandardized["voted2012"] = df_unstandardized["voted2012"].replace(2, 0)

df_unstandardized["voted2016"] = df_unstandardized["voted2016"].where(
    ~df_unstandardized["voted2016"].isin([-9, -8, -1]), np.nan
)
df_unstandardized["voted2016"] = df_unstandardized["voted2016"].replace(2, 0)

df_unstandardized["voted_for_2012"] = df_unstandardized["voted_for_2012"].where(
    ~df_unstandardized["voted_for_2012"].isin([-9, -8, -1, 5]), np.nan
)
df_unstandardized["voted_for_2016"] = df_unstandardized["voted_for_2016"].where(
    ~df_unstandardized["voted_for_2016"].isin([-9, -8, -1, 5]), np.nan
)

df_unstandardized["expectations_2020"] = df_unstandardized["expectations_2020"].where(
    ~df_unstandardized["expectations_2020"].isin([-9, -8, 5]), np.nan
)
df_unstandardized["expectations_2020"] = df_unstandardized["expectations_2020"].replace(2, 3)

# --- Feeling thermometers ---
df_unstandardized["feeling_dems"] = df_unstandardized["feeling_dems"].where(
    ~df_unstandardized["feeling_dems"].isin([-9, 998]), np.nan
)
df_unstandardized["feeling_reps"] = df_unstandardized["feeling_reps"].where(
    ~df_unstandardized["feeling_reps"].isin([-9, 998]), np.nan
)

# --- Main index items (higher = more liberal) ---
for col in ["free_press", "checks_and_balances", "rule_of_law", "agree_on_facts"]:
    df_unstandardized[col] = df_unstandardized[col].where(
        ~df_unstandardized[col].isin([-9, -8]), np.nan
    )

df_unstandardized["unitary_executive"] = df_unstandardized["unitary_executive"].where(
    df_unstandardized["unitary_executive"] != -2, np.nan
)
df_unstandardized["journalist_access"] = df_unstandardized["journalist_access"].where(
    df_unstandardized["journalist_access"] != -2, np.nan
)
df_unstandardized["media_undermined_concern"] = df_unstandardized[
    "media_undermined_concern"
].where(~df_unstandardized["media_undermined_concern"].isin([-9, -8]), np.nan)

# --- Other items ---
df_unstandardized["media_trust"] = df_unstandardized["media_trust"].where(
    ~df_unstandardized["media_trust"].isin([-9, -8]), np.nan
)
df_unstandardized["govt_trust"] = df_unstandardized["govt_trust"].where(
    ~df_unstandardized["govt_trust"].isin([-9, -8]), np.nan
)
df_unstandardized["govt_trust"] = 6 - df_unstandardized["govt_trust"]  # reverse-coded
df_unstandardized["govt_capture"] = df_unstandardized["govt_capture"].where(
    ~df_unstandardized["govt_capture"].isin([-9, -8]), np.nan
)
df_unstandardized["govt_corruption"] = df_unstandardized["govt_corruption"].where(
    ~df_unstandardized["govt_corruption"].isin([-9, -8]), np.nan
)
df_unstandardized["urban_unrest"] = df_unstandardized["urban_unrest"].where(
    ~df_unstandardized["urban_unrest"].isin([-9, -8, 99]), np.nan
)
df_unstandardized["urban_unrest"] = 8 - df_unstandardized["urban_unrest"]  # reverse-coded
df_unstandardized["foreign_help"] = df_unstandardized["foreign_help"].where(
    ~df_unstandardized["foreign_help"].isin([-9, -8]), np.nan
)
df_unstandardized["govt_principled"] = df_unstandardized["govt_principled"].where(
    ~df_unstandardized["govt_principled"].isin([-9, -8]), np.nan
)

##############################################################################
# SECTION 3: CREATE TREATMENT VARIABLE
##############################################################################

# NOTE: In the restricted dataset, actual dates of birth are available.
# Here we simulate month and day (as in the R script practice version)
# so the code runs before access to restricted DOB data.

np.random.seed(42)
n = len(df_unstandardized)

df_unstandardized["year_of_birth"] = (2020 - df_unstandardized["age"]).astype("Int64")
df_unstandardized["month_of_birth"] = np.random.randint(1, 13, size=n)
df_unstandardized["day_of_birth"] = np.random.randint(1, 29, size=n)


def make_date(row):
    try:
        return pd.Timestamp(
            int(row["year_of_birth"]), int(row["month_of_birth"]), int(row["day_of_birth"])
        )
    except Exception:
        return pd.NaT


df_unstandardized["date_of_birth"] = df_unstandardized.apply(make_date, axis=1)

date_cutoff = pd.Timestamp("1994-11-06")
date_2012_election = pd.Timestamp("2012-11-06")

df_unstandardized["days_from_cutoff"] = (
    df_unstandardized["date_of_birth"] - date_cutoff
).dt.days.astype(float)

df_unstandardized["age_2012_election"] = (
    date_2012_election - df_unstandardized["date_of_birth"]
).dt.days / 365.25

df_unstandardized["treatment"] = (
    df_unstandardized["date_of_birth"] > date_cutoff
).astype(int)

##############################################################################
# SECTION 4: INDEX CONSTRUCTION
##############################################################################

liberal_items = [
    "free_press",
    "checks_and_balances",
    "rule_of_law",
    "agree_on_facts",
    "unitary_executive",
    "journalist_access",
    "media_undermined_concern",
]

# --- Standardize items (z-score) ---
df = df_unstandardized.copy()
for item in liberal_items:
    col = df[item].astype(float)
    df[item] = (col - col.mean()) / col.std()


# --- Cronbach's Alpha ---
def cronbach_alpha(data, items):
    df_items = data[items].dropna().astype(float)
    k = len(items)
    item_vars = df_items.var(axis=0, ddof=1).sum()
    total_var = df_items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars / total_var)


raw_alpha = cronbach_alpha(df_unstandardized, liberal_items)
std_alpha = cronbach_alpha(df, liberal_items)
print(f"Cronbach's Alpha (raw): {raw_alpha:.3f}")
print(f"Cronbach's Alpha (std): {std_alpha:.3f}")

alpha_summary = pd.DataFrame(
    {
        "metric": ["Cronbach's Alpha (raw)", "Cronbach's Alpha (std)", "n_items"],
        "value": [round(raw_alpha, 3), round(std_alpha, 3), len(liberal_items)],
    }
)
alpha_summary.to_csv(
    os.path.join(results_path, "Appendix/Index/fa_alpha_summary.csv"), index=False
)


# --- Item-total correlations (drop from composite before correlating) ---
def item_total_correlations(data, items):
    df_items = data[items].dropna().astype(float)
    out = {}
    for item in items:
        rest_sum = df_items.drop(columns=[item]).sum(axis=1)
        r, _ = stats.pearsonr(df_items[item], rest_sum)
        out[item] = r
    return out


item_total = item_total_correlations(df_unstandardized, liberal_items)


# --- Factor Analysis (1 factor, principal axis) ---
fa_data = df_unstandardized[liberal_items].dropna().astype(float)
fa = FactorAnalyzer(n_factors=1, rotation=None, method="principal")
fa.fit(fa_data)

loadings = fa.loadings_[:, 0]
communalities = fa.get_communalities()
uniquenesses = fa.get_uniquenesses()
eigenvalues, _ = fa.get_eigenvalues()

fa_diagnostics = pd.DataFrame(
    {
        "item": liberal_items,
        "loading": loadings.round(3),
        "communality": communalities.round(3),
        "uniqueness": uniquenesses.round(3),
        "item_total_r": [round(item_total[i], 3) for i in liberal_items],
    }
)
fa_diagnostics.to_csv(
    os.path.join(results_path, "Appendix/Index/fa_diagnostics.csv"), index=False
)
print(fa_diagnostics)

fa_eigenvalues_df = pd.DataFrame(
    {
        "factor": range(1, len(eigenvalues) + 1),
        "eigenvalue": eigenvalues.round(3),
        "variance_explained": (eigenvalues / eigenvalues.sum()).round(3),
        "cumulative_variance": (eigenvalues.cumsum() / eigenvalues.sum()).round(3),
    }
)
fa_eigenvalues_df.to_csv(
    os.path.join(results_path, "Appendix/Index/fa_eigenvalues.csv"), index=False
)


# --- Scree plot ---
def get_screeplot(eigenvalues, title="Scree Plot"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, "o-", markersize=5)
    ax.axhline(y=1, linestyle="--", color="gray")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(1, len(eigenvalues) + 1))
    plt.tight_layout()
    return fig


fig_scree = get_screeplot(eigenvalues, "Scree Plot: Liberalism Index")
fig_scree.savefig(
    os.path.join(results_path, "Appendix/Index/screeplot_liberal.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# --- Build index from factor scores, rescaled to [0, 1] ---
df_items_for_scoring = df[liberal_items].astype(float)
valid_mask = df_items_for_scoring.notna().all(axis=1)

fa_scores = np.full(len(df), np.nan)
fa_scores[valid_mask.values] = fa.transform(df_items_for_scoring[valid_mask])[:, 0]

valid_scores = fa_scores[~np.isnan(fa_scores)]
score_min, score_max = valid_scores.min(), valid_scores.max()
df["liberal_index"] = np.where(
    ~np.isnan(fa_scores),
    (fa_scores - score_min) / (score_max - score_min),
    np.nan,
)

##############################################################################
# SECTION 5: SUBGROUP DEFINITIONS AND DISCONTINUITY PLOTS
##############################################################################

# H1: Full sample
# H2: Independents vs. Partisans (combined Democrats + Republicans)
# H3: Democrats vs. Republicans separately

df_democrats = df[df["party_summary"] == "Democrat"].copy()
df_republicans = df[df["party_summary"] == "Republican"].copy()
df_partisans = df[df["party_summary"].isin(["Democrat", "Republican"])].copy()
df_independents = df[df["party_summary"] == "Independent"].copy()


def get_discontinuityplot(
    dataframe,
    outcome="liberal_index",
    outcome_name="Liberal Norm Support",
    party_id="party_summary",
):
    df_plot = dataframe[~dataframe[party_id].isna()].copy()
    df_plot = df_plot[df_plot["days_from_cutoff"] >= -4000]

    party_label_map = {
        "Democrat": "Democrats",
        "Republican": "Republicans",
        "Independent": "Independents",
    }
    df_plot["subgroup"] = df_plot[party_id].map(party_label_map)

    df_partisans_plot = df_plot[df_plot["subgroup"].isin(["Democrats", "Republicans"])].copy()
    df_partisans_plot["subgroup"] = "Partisans"
    df_combined = pd.concat([df_plot, df_partisans_plot], ignore_index=True)

    subgroup_order = ["Independents", "Partisans", "Democrats", "Republicans"]
    df_combined = df_combined[
        df_combined[outcome].notna() & df_combined["subgroup"].notna()
    ]
    df_combined["subgroup"] = pd.Categorical(
        df_combined["subgroup"], categories=subgroup_order, ordered=True
    )

    colors = {0: "#1b9e77", 1: "#d95f02"}

    # Full-sample plot
    fig_all, ax_all = plt.subplots(figsize=(6.5, 5))
    full_valid = dataframe[dataframe[outcome].notna()]
    for treat_val, group in full_valid.groupby("treatment"):
        ax_all.scatter(
            group["age_2012_election"],
            group[outcome],
            alpha=0.3,
            s=10,
            color=colors[treat_val],
            label="Eligible" if treat_val == 0 else "Not Eligible",
        )
        xv = group["age_2012_election"].values
        yv = group[outcome].values
        mask = ~(np.isnan(xv) | np.isnan(yv))
        if mask.sum() > 2:
            z = np.polyfit(xv[mask], yv[mask], 1)
            x_range = np.linspace(xv[mask].min(), xv[mask].max(), 200)
            ax_all.plot(x_range, np.polyval(z, x_range), color="black", linewidth=1.5)
    ax_all.axvline(x=18, linestyle="--", color="black")
    ax_all.set_xlabel("Age at 2012 Election")
    ax_all.set_ylabel(f"{outcome_name} Index (0–1)")
    ax_all.set_title(
        f"Discontinuity in {outcome_name}\nIndependents, Democrats, and Republicans (Full Sample)",
        fontweight="bold",
    )
    ax_all.legend(title="2012 Voting Eligibility", loc="lower right")
    plt.tight_layout()

    # Subgroup plot (2×2 grid)
    fig_sub, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))
    axes = axes.flatten()
    for idx, subgroup in enumerate(subgroup_order):
        ax = axes[idx]
        sub_data = df_combined[df_combined["subgroup"] == subgroup]
        for treat_val, group in sub_data.groupby("treatment"):
            ax.scatter(
                group["age_2012_election"],
                group[outcome],
                alpha=0.3,
                s=8,
                color=colors[treat_val],
            )
            xv = group["age_2012_election"].values
            yv = group[outcome].values
            mask = ~(np.isnan(xv) | np.isnan(yv))
            if mask.sum() > 2:
                z = np.polyfit(xv[mask], yv[mask], 1)
                x_range = np.linspace(xv[mask].min(), xv[mask].max(), 200)
                ax.plot(x_range, np.polyval(z, x_range), color="black", linewidth=1.2)
        ax.axvline(x=18, linestyle="--", color="black", linewidth=0.8)
        ax.set_title(subgroup)
        ax.set_xlabel("Age at 2012 Election")
        ax.set_ylabel(f"{outcome_name} Index")

    handles = [
        mpatches.Patch(color=colors[0], label="Eligible"),
        mpatches.Patch(color=colors[1], label="Not Eligible"),
    ]
    fig_sub.legend(
        handles=handles,
        title="2012 Voting Eligibility",
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig_sub.suptitle(
        f"Discontinuity in {outcome_name}\nAmong Different Partisanship Categories",
        fontweight="bold",
    )
    plt.tight_layout()

    return {"plot_all": fig_all, "plot_subgroups": fig_sub}


disc_plots = get_discontinuityplot(df)
disc_plots["plot_all"].savefig(
    os.path.join(results_path, "Main/Plots/Discontinuity/discontinuityplot_full.png"),
    dpi=300,
    bbox_inches="tight",
)
disc_plots["plot_subgroups"].savefig(
    os.path.join(results_path, "Main/Plots/Discontinuity/discontinuityplot_subgroups.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

##############################################################################
# SECTION 6: COVARIATE BALANCE AND CONTROL SELECTION
##############################################################################


def get_imbalanced_controls(data, potential_controls, thr_num=0.3, thr_cat=0.05):
    imbalanced = []
    for var in potential_controls:
        col_nonmissing = data[var].dropna()
        idx = col_nonmissing.index
        treat = data.loc[idx, "treatment"]

        if pd.api.types.is_numeric_dtype(data[var]) and data[var].nunique() > 5:
            g0 = data.loc[data["treatment"] == 0, var].dropna()
            g1 = data.loc[data["treatment"] == 1, var].dropna()
            _, pval = ttest_ind(g0, g1)
            threshold = thr_num
        else:
            ct = pd.crosstab(data[var].fillna("Missing"), data["treatment"])
            _, pval, _, _ = chi2_contingency(ct)
            threshold = thr_cat

        if pval < threshold:
            imbalanced.append(var)
            print(f"  {var}: IMBALANCED (p={pval:.3f})")
        else:
            print(f"  {var}: balanced (p={pval:.3f})")
    return imbalanced


potential_controls = ["education", "sex", "income", "race"]
imbalanced_controls = get_imbalanced_controls(df, potential_controls)


# --- Race dummies (White as baseline, dropped) ---
def add_race_dummies(data):
    dummies = pd.get_dummies(data["race"], prefix="race", drop_first=False)
    if "race_White" in dummies.columns:
        dummies = dummies.drop(columns=["race_White"])
    return pd.concat([data, dummies.astype(float)], axis=1)


df = add_race_dummies(df)
df_democrats = add_race_dummies(df_democrats)
df_republicans = add_race_dummies(df_republicans)
df_independents = add_race_dummies(df_independents)
df_partisans = add_race_dummies(df_partisans)

race_dummies = ["race_Black", "race_Hispanic", "race_Asian", "race_Native", "race_Other"]
controls = [c for c in imbalanced_controls if c != "race"] + race_dummies
print(f"\nFinal controls used: {controls}")

##############################################################################
# SECTION 7: RDD ANALYSIS — HELPER FUNCTIONS
##############################################################################


def pull_estimates(r):
    return {
        "coef_conv": float(r.coef[0]),
        "se_conv": float(r.se[0]),
        "pv_conv": float(r.pv[0]),
        "coef_bc": float(r.coef[1]),
        "se_bc": float(r.se[1]),
        "pv_bc": float(r.pv[1]),
        "coef_rob": float(r.coef[2]),
        "se_rob": float(r.se[2]),
        "pv_rob": float(r.pv[2]),
    }


def extract_rdd_summary(rd_object, model_label="Model"):
    est = pull_estimates(rd_object)
    bwselect = getattr(rd_object, "bwselect", "mserd")
    bw_type = "MSE-optimal" if bwselect == "mserd" else bwselect
    bw_h = round(float(rd_object.bws[0]), 2)
    n_h = int(rd_object.N_h[0]) + int(rd_object.N_h[1])

    rows = []
    for est_type, ck, sk, pk in [
        ("Conventional", "coef_conv", "se_conv", "pv_conv"),
        ("Bias-Corrected", "coef_bc", "se_bc", "pv_bc"),
        ("Bias-Corrected (Robust SE)", "coef_rob", "se_rob", "pv_rob"),
    ]:
        rows.append(
            {
                "Model": model_label,
                "Estimate Type": est_type,
                "Estimate": round(est[ck], 3),
                "SE": round(est[sk], 3),
                "P-Value": round(est[pk], 3),
                "Bandwidth Type": bw_type,
                "Bandwidth (h)": bw_h,
                "N": n_h,
            }
        )
    return pd.DataFrame(rows)


def _prep_arrays(data, index_var, controls):
    """Return (y, x, covs_df) with fully valid rows."""
    y = data[index_var].values.astype(float)
    x = data["days_from_cutoff"].values.astype(float)
    valid = ~(np.isnan(y) | np.isnan(x))
    y, x = y[valid], x[valid]
    covs_df = data[controls].iloc[np.where(valid)[0]].astype(float)
    return y, x, covs_df


def run_rdd_models(data, index_var, controls, sample_label):
    y, x, covs_df = _prep_arrays(data, index_var, controls)

    # Without controls
    rdd_simple = rdrobust(y=y, x=x, c=0, all=True)
    summary_simple = extract_rdd_summary(rdd_simple, "Without Controls")

    # With controls — drop rows that have any missing covariate
    valid2 = covs_df.notna().all(axis=1).values
    y2, x2 = y[valid2], x[valid2]
    covs_arr = covs_df.iloc[np.where(valid2)[0]].values

    rdd_controls = rdrobust(y=y2, x=x2, c=0, covs=covs_arr, all=True)
    summary_controls = extract_rdd_summary(rdd_controls, "With Controls")

    result = pd.concat([summary_simple, summary_controls], ignore_index=True)
    result.insert(0, "Sample", sample_label)
    return result


##############################################################################
# SECTION 8: RUN RDD MODELS
##############################################################################

# H1 — Full sample effect (does first-time voting exposure shift liberal norm support?)
rdd_liberal_full = run_rdd_models(df, "liberal_index", controls, "Full Sample")

# H2 — Effect differs for independents vs. partisans
rdd_liberal_independents = run_rdd_models(
    df_independents, "liberal_index", controls, "Independents"
)
rdd_liberal_partisans = run_rdd_models(
    df_partisans, "liberal_index", controls, "Partisans"
)

# H3 — Partisan heterogeneity (Democrats vs. Republicans)
rdd_liberal_democrats = run_rdd_models(
    df_democrats, "liberal_index", controls, "Democrats"
)
rdd_liberal_republicans = run_rdd_models(
    df_republicans, "liberal_index", controls, "Republicans"
)

# Combine and export
sample_order = ["Republicans", "Democrats", "Partisans", "Independents", "Full Sample"]

rdd_liberal = pd.concat(
    [
        rdd_liberal_full,
        rdd_liberal_independents,
        rdd_liberal_partisans,
        rdd_liberal_democrats,
        rdd_liberal_republicans,
    ],
    ignore_index=True,
)
rdd_liberal["Outcome"] = "Liberal Attitudes"
rdd_liberal["Sample"] = pd.Categorical(
    rdd_liberal["Sample"], categories=sample_order, ordered=True
)

rdd_liberal.to_csv(os.path.join(results_path, "Main/rdd_liberal.csv"), index=False)
print(rdd_liberal)

##############################################################################
# SECTION 9: COEFFICIENT PLOTS
##############################################################################


def get_coefplot(
    dataframe,
    title="Estimated Effects Across Subgroups",
    subtitle="Bias-Adjusted Estimates",
):
    colors = {"Without Controls": "#20b2aa", "With Controls": "#8b1a8b"}
    offsets = {"Without Controls": 0.15, "With Controls": -0.15}

    samples = list(reversed(dataframe["Sample"].cat.categories))
    y_map = {s: i for i, s in enumerate(samples)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in dataframe.groupby("Model"):
        for _, row in group.iterrows():
            y = y_map[row["Sample"]] + offsets.get(model, 0)
            ax.scatter(row["Estimate"], y, color=colors.get(model, "blue"), s=50, zorder=5)
            ax.errorbar(
                row["Estimate"],
                y,
                xerr=1.96 * row["SE"],
                fmt="none",
                color=colors.get(model, "blue"),
                linewidth=1.5,
                capsize=3,
            )

    ax.axvline(x=0, linestyle="--", color="gray", alpha=0.7)
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Estimated LATE (τ)")
    ax.set_title(f"{title}\n{subtitle}", fontweight="bold")

    handles = [
        mlines.Line2D([], [], color=colors[m], marker="o", linestyle="-", label=m)
        for m in ["Without Controls", "With Controls"]
    ]
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    return fig


def get_coefplot_robustness(
    dataframe,
    title="Estimated Effects Across Subgroups",
    subtitle="Across Different Model Specifications",
):
    colors = {
        "Conventional": "gray",
        "Bias-Corrected": "olivedrab",
        "Bias-Corrected (Robust SE)": "#cd853f",
    }
    offsets = {
        "Conventional": 0.2,
        "Bias-Corrected": 0.0,
        "Bias-Corrected (Robust SE)": -0.2,
    }

    models = dataframe["Model"].unique()
    samples = list(reversed(dataframe["Sample"].cat.categories))
    y_map = {s: i for i, s in enumerate(samples)}

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_data = dataframe[dataframe["Model"] == model]
        for est_type, group in model_data.groupby("Estimate Type"):
            for _, row in group.iterrows():
                y = y_map[row["Sample"]] + offsets.get(est_type, 0)
                ax.scatter(
                    row["Estimate"],
                    y,
                    color=colors.get(est_type, "blue"),
                    s=40,
                    zorder=5,
                )
                ax.errorbar(
                    row["Estimate"],
                    y,
                    xerr=1.96 * row["SE"],
                    fmt="none",
                    color=colors.get(est_type, "blue"),
                    linewidth=1.5,
                    capsize=3,
                )
        ax.axvline(x=0, linestyle="--", color="gray", alpha=0.7)
        ax.set_yticks(list(y_map.values()))
        ax.set_yticklabels(list(y_map.keys()))
        ax.set_xlabel("Estimated LATE (τ)")
        ax.set_title(model, fontweight="bold")

    handles = [
        mlines.Line2D([], [], color=colors[e], marker="o", linestyle="-", label=e)
        for e in colors
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{title}\n{subtitle}", fontweight="bold")
    plt.tight_layout()
    return fig


# Bias-corrected estimates for the clean coefplot
rdd_bc = rdd_liberal[rdd_liberal["Estimate Type"] == "Bias-Corrected"].copy()
rdd_bc["Model"] = pd.Categorical(
    rdd_bc["Model"], categories=["Without Controls", "With Controls"], ordered=True
)
rdd_bc["Sample"] = pd.Categorical(rdd_bc["Sample"], categories=sample_order, ordered=True)

fig_coef = get_coefplot(rdd_bc)
fig_coef.savefig(
    os.path.join(results_path, "Main/Plots/coefplot_liberal.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# All estimate types for the robustness coefplot
rdd_all_types = rdd_liberal.copy()
rdd_all_types["Estimate Type"] = pd.Categorical(
    rdd_all_types["Estimate Type"],
    categories=["Conventional", "Bias-Corrected", "Bias-Corrected (Robust SE)"],
    ordered=True,
)
rdd_all_types["Sample"] = pd.Categorical(
    rdd_all_types["Sample"], categories=sample_order, ordered=True
)

fig_coef_rob = get_coefplot_robustness(rdd_all_types)
fig_coef_rob.savefig(
    os.path.join(results_path, "Main/Plots/coefplot_robustness.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

##############################################################################
# SECTION 10: ROBUSTNESS CHECKS
##############################################################################


def run_robustness_checks(data, index_var, controls, sample_label):
    y, x, covs_df = _prep_arrays(data, index_var, controls)

    # Optimal bandwidth (with controls) used to set the sensitivity range
    valid_covs = covs_df.notna().all(axis=1).values
    y_c, x_c = y[valid_covs], x[valid_covs]
    covs_c = covs_df.iloc[np.where(valid_covs)[0]].values
    h_opt = float(rdrobust(y=y_c, x=x_c, c=0, covs=covs_c).bws[0])

    all_rows = []

    # --- Bandwidth sensitivity ---
    bws = np.unique(np.round(np.linspace(h_opt * 0.3, h_opt * 2.0, 20)))
    for use_covs in [False, True]:
        yy, xx, cc = (y_c, x_c, covs_c) if use_covs else (y, x, None)
        for h in bws:
            try:
                r = rdrobust(y=yy, x=xx, c=0, h=float(h), covs=cc, all=True)
                row = pull_estimates(r)
                row.update(
                    {
                        "bandwidth": h,
                        "controls": use_covs,
                        "h_opt": h_opt,
                        "check_type": "bandwidth",
                        "sample": sample_label,
                    }
                )
                all_rows.append(row)
            except Exception:
                pass

    # --- Placebo cutoffs ---
    placebo_cutoffs = [-730, -365, 0, 365, 730]
    for use_covs in [False, True]:
        yy, xx, cc = (y_c, x_c, covs_c) if use_covs else (y, x, None)
        for co in placebo_cutoffs:
            try:
                r = rdrobust(y=yy, x=xx, c=co, covs=cc, all=True)
                row = pull_estimates(r)
                row.update(
                    {
                        "cutoff": co,
                        "controls": use_covs,
                        "check_type": "cutoffs",
                        "sample": sample_label,
                    }
                )
                all_rows.append(row)
            except Exception:
                pass

    # --- Polynomial degree ---
    for use_covs in [False, True]:
        yy, xx, cc = (y_c, x_c, covs_c) if use_covs else (y, x, None)
        for p in [1, 2, 3]:
            try:
                r = rdrobust(y=yy, x=xx, c=0, p=p, covs=cc, all=True)
                row = pull_estimates(r)
                row.update(
                    {
                        "polynomial": p,
                        "controls": use_covs,
                        "check_type": "polynomial",
                        "sample": sample_label,
                    }
                )
                all_rows.append(row)
            except Exception:
                pass

    return pd.DataFrame(all_rows)


robustness_subgroups = [
    (df, "Full Sample"),
    (df_independents, "Independents"),
    (df_partisans, "Partisans"),
    (df_democrats, "Democrats"),
    (df_republicans, "Republicans"),
]

all_robustness_dfs = []
for data, label in robustness_subgroups:
    print(f"Running robustness checks: {label}")
    rob = run_robustness_checks(data, "liberal_index", controls, label)
    all_robustness_dfs.append(rob)

all_robustness = pd.concat(all_robustness_dfs, ignore_index=True)
all_robustness.to_csv(
    os.path.join(
        results_path, "Appendix/Robustness Checks/all_robustness_checks.csv"
    ),
    index=False,
)


# --- Robustness plots ---
def get_robustnessplots(data, subgroup_name, estimate_type=3, include_covariates=True):
    """
    estimate_type: 1=Conventional, 2=Bias-Corrected, 3=BC+Robust SE (preferred)
    """
    cols = {
        1: ("coef_conv", "se_conv", "Conventional"),
        2: ("coef_bc", "se_bc", "Bias-Corrected"),
        3: ("coef_rob", "se_rob", "Bias-Corrected (Robust SE)"),
    }
    coef_col, se_col, est_name = cols[estimate_type]
    control_label = "(With Controls)" if include_covariates else "(Without Controls)"

    def prep(check_type):
        return data[
            (data["check_type"] == check_type)
            & (data["sample"] == subgroup_name)
            & (data["controls"] == include_covariates)
        ].copy()

    bw_plot = prep("bandwidth")
    co_plot = prep("cutoffs")
    poly_plot = prep("polynomial")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    def make_panel(ax, df_panel, x_var, xlabel, title):
        if df_panel.empty:
            ax.set_title(title)
            return
        ax.scatter(df_panel[x_var], df_panel[coef_col], color="black", s=20, zorder=5)
        ax.errorbar(
            df_panel[x_var],
            df_panel[coef_col],
            yerr=1.96 * df_panel[se_col],
            fmt="none",
            color="black",
            linewidth=0.8,
            capsize=0,
        )
        ax.axhline(y=0, linestyle="--", color="gray")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("LATE", fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.tick_params(labelsize=7)

    h_opt_val = bw_plot["h_opt"].iloc[0] if not bw_plot.empty else None
    make_panel(axes[0], bw_plot, "bandwidth", "Bandwidth (days)", "Bandwidth Sensitivity")
    if h_opt_val:
        axes[0].axvline(x=h_opt_val, linestyle="--", color="gray", linewidth=0.8)

    make_panel(axes[1], co_plot, "cutoff", "Placebo Cutoff (days)", "Placebo Cutoffs")
    make_panel(axes[2], poly_plot, "polynomial", "Polynomial Degree", "Polynomial Degree")
    if not poly_plot.empty:
        axes[2].set_xticks([1, 2, 3])

    fig.suptitle(
        f"{est_name} — {subgroup_name} {control_label}", fontweight="bold", fontsize=10
    )
    plt.tight_layout()
    return fig


# Generate and display all robustness plots
subgroup_strings = ["Full Sample", "Independents", "Partisans", "Democrats", "Republicans"]
for sg in subgroup_strings:
    fig_rob = get_robustnessplots(
        all_robustness, sg, estimate_type=2, include_covariates=True
    )
    fig_rob.savefig(
        os.path.join(
            results_path, f"Appendix/Robustness Checks/robustness_{sg.replace(' ', '_')}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
