import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn import config_context
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2


X, y = load_iris(return_X_y=True, as_frame=True)
X = X.astype({"petal width (cm)": np.float16, "petal length (cm)": np.float16})
X["cat"] = y.astype("category")
selector = SelectKBest(chi2, k=2).set_output(transform="pandas")
output = selector.fit_transform(X, y)
assert_frame_equal(output, X.loc[:, selector.get_support()])
selector.set_output(transform="default")
assert isinstance(selector.transform(X), np.ndarray)

selector = SelectKBest(chi2, k=0).fit(X, y)
X["cat"] = pd.Categorical(y, ordered=True)
X["nullable"] = pd.array(y, dtype="Int64")
selector = SelectKBest(chi2, k="all").fit(X, y)
X["nullable"] = X["nullable"].mask(X.index == 0)
X["cat"] = pd.Categorical(["a", "b", "c"] * 50, ordered=True)
X.index = pd.Index(range(150, 300), name="row")
with config_context(transform_output="pandas"):
    assert_frame_equal(selector.transform(X), X)
    selector.k = 0
    assert_frame_equal(selector.transform(X), X.iloc[:, :0])
    selector.k = "all"
    try:
        selector.transform(X.iloc[:, :-1])
    except ValueError:
        pass
    else:
        raise AssertionError("missing feature validation")

print("dtype, index, empty output, configuration and feature validation checks passed")

