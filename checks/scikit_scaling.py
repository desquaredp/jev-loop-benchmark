import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


frame = pd.DataFrame({"x": [0, 1, 2]}, dtype="int64")
actual = StandardScaler().set_output(transform="pandas").fit_transform(frame)
expected = np.array([-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
assert np.allclose(actual.x.to_numpy(), expected), (
    "Scaling must not truncate fractional values to the input integer dtype"
)
