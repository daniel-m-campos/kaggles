# %%
import glob

#  %%
import house_price_utils as hpu
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from ml_tools.experiment import RegressionExperiment

# %%
lm_models = {m: RegressionExperiment.load(m) for m in glob.glob("../models/*.pkl")}

# %%
scores = {k: v.scores for k, v in lm_models.items()}
summary = hpu.summary(scores)
summary

# %% Select the best model for each class
ensemble_models = {}
for row in summary.head(4).iterrows():
    model_class = row[0].split("_")[0].split("/")[-1]
    if model_class not in ensemble_models:
        ensemble_models[model_class] = lm_models[row[0]]
ensemble_models

# %%
data, data_test = hpu.load_data()
X, y = hpu.extract_target(data, "SalePrice")

# %%
predictions = pd.DataFrame(
    {k: v.oof_predictions for k, v in ensemble_models.items()}, index=X.index
)
predictions = predictions.join(y)

# %%
g = sns.clustermap(
    predictions.corr(),
    method="complete",
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 7},
    vmin=-1,
    vmax=1,
    figsize=(15, 10),
)

sns.pairplot(predictions, diag_kind="kde")

# %%
predictions = predictions.drop(columns="SalePrice")


# %%
ensemble = predictions.mean(axis=1)
ax = hpu.residual_plots(y, ensemble)

# # %%
ensemble = make_pipeline(
    linear_model.LinearRegression(),
)

scores = cross_val_score(
    ensemble, predictions, y, cv=5, scoring="neg_mean_absolute_error"
)
-scores.mean(), scores.std()

# %%
_ = ensemble.fit(predictions, y)

pd.DataFrame(
    ensemble[0].coef_, index=predictions.columns, columns=["coef"]
).sort_values("coef", ascending=False)

# %% Ensemble test prediction

test_predictions = pd.DataFrame(
    {k: v.fit(X, y).predict(data_test) for k, v in ensemble_models.items()},
    index=data_test.index,
)

# %%
output = pd.DataFrame(
    {
        "Id": data_test.index,
        "SalePrice": ensemble.predict(test_predictions),
    }
)
output.to_csv("submission.csv", index=False)


# %%
SUBMISSION = True

if SUBMISSION:
    import kaggle

    desc = f"Linear regression ensemble of {', '.join(ensemble_models.keys())} "
    "full data fit"
    for competition in (
        "home-data-for-ml-course",
        "house-prices-advanced-regression-techniques",
    ):
        result = kaggle.api.competition_submit(
            "submission.csv",
            desc,
            competition,
        )
        print(result)

# %%
