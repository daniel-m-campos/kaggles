# %%
import glob

import house_price_utils as hpu

from ml_tools.experiment import RegressionExperiment

# %%
lm_models = {m: RegressionExperiment.load(m) for m in glob.glob("../models/*.pkl")}

# %%
scores = {k: v.scores for k, v in lm_models.items()}

# %%
summary = hpu.summary(scores)
summary

# %%
