import pandas as pd
import numpy as np


class PNN:
    def __init__(self, df: pd.DataFrame, sigma: float=0.5):
        """
        DataFrame info:
            1. Must contain target column named "target"
            2. All other columns are considered features
        """
        assert "target" in df.columns, "No target column named 'target' found!"
        assert len(df.columns) > 1, "Not enough feature columns in dataframe!"

        df = df.copy()
        self.target = df.pop("target")
        self.features = df

        check = [df[f'{col}'].dtype in ("float64", "int64") for col in df.columns]
        assert all(check), "All feature columns must be type 'float64' or 'int64'!"
        
        self.num_features = len(self.features.columns)

        self.input_layer = InputLayer(self.num_features)
        self.pattern_layer = PatternLayer(self.features, sigma=sigma)
        self.summation_layer = SummationLayer(self.target)
        self.decision_layer = DecisionLayer()
    
    def __call__(self, features: np.ndarray) -> str:
        pattern_out = self.pattern_layer(self.input_layer(features))
        summation_out = self.summation_layer(pattern_out, self.target)
        decision_out = self.decision_layer(summation_out)

        return decision_out

class InputLayer:
    def __init__(self, num_features: int):
        self.num_features = num_features
    
    def __call__(self, features):
        assert len(features) == self.num_features, f"Number of training features != {self.num_features}"
        return features


class PatternLayer:
    def __init__(self, df, sigma):
        self.num_features = len(df.columns)
        self.num_samples = len(df)

        self.sigma_square = sigma**2

        self.weights = []
        for i in range(self.num_samples):
            self.weights.append(df.iloc[i].values)
        
        self.weights = np.array(self.weights)
    
    def __call__(self, features: np.ndarray):
        shape = (self.num_features,)
        assert features.shape == shape, f"Feature array must be of shape {shape}!"

        outs = []
        for i in range(len(self.weights)):
            euclidean_distance = np.sum((self.weights[i]-features)**2)
            outs.append(np.exp(-euclidean_distance/self.sigma_square))

        return np.array([outs]).T


class SummationLayer:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, inputs, target):
        df = target.to_frame()
        df['pattern_outs'] = inputs

        outs = df.groupby("target").sum()
        return outs


class DecisionLayer:
    def __init__(self):
        pass
    
    def __call__(self, summation_out: pd.DataFrame):
        highest_scoring_class_idx = np.argmax(summation_out)
        classified_class = summation_out.index[highest_scoring_class_idx]

        return classified_class
