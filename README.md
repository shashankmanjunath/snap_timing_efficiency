# Snap Timing Efficiency: A Novel Metric to Analyze how Efficiently Teams use Snap Timing in Conjunction with Pre-Snap Motion

Author: [Shashank Manjunath](https://shashankmanjunath.github.io/)

This repository contains code used in the 2025 NFL Big Data Bowl submission Snap
Timing Efficiency.

The official Kaggle notebook can be viewed [at this link](https://www.kaggle.com/code/tankshank/snap-timing-efficiency).

# Running the code

In order to run the code, first please install the requirements included in the
`requirements.txt` file using your favorite package manager. For example, using
anaconda, a new environment can be created using the following command:

```
conda create --name nfl_bdb_2025_ste --file requirements.txt
```

Once this is run, [download the NFL BDB 2025
data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data) and place
it into a directory, henceforth referred to as `${DATA_DIR}`. To train the
XGBoost model, run the following command:

```
python xgb_model.py train --data_dir ${DATA_DIR} --route_type all
```

This will first preprocess the data, which takes ~30 minutes, and save it into a
cache file called `cache_dataset.hdf5` in `${DATA_DIR}`. It will then
automatically train an XGBoost model on data from weeks 1-7 and test it on data
from weeks 8 and 9, returning performance metrics.

# Recreating the Analysis

To recreate the analysis, start a jupyter lab session and run the code included
in `ste_analysis.ipynb`
