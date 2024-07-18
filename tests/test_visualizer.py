# tests/test_visualizer.py

import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from da_vis.visualizer import DataVisualizer  

class TestDataVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_iris()
        cls.df = pd.DataFrame(data.data, columns=data.feature_names)
        cls.model = RandomForestClassifier()
        cls.model.fit(cls.df, data.target)
        cls.visualizer = DataVisualizer(data=cls.df, model=cls.model)
    
    def test_correlation_heatmap(self):
        try:
            self.visualizer.correlation_heatmap()
        except Exception as e:
            self.fail(f'correlation_heatmap() raised {e} unexpectedly!')

    def test_feature_distribution(self):
        try:
            self.visualizer.feature_distribution('sepal length (cm)')
        except Exception as e:
            self.fail(f'feature_distribution() raised {e} unexpectedly!')

    def test_confusion_matrix(self):
        try:
            self.visualizer.confusion_matrix(self.model.predict(self.df), self.model.predict(self.df))
        except Exception as e:
            self.fail(f'confusion_matrix() raised {e} unexpectedly!')

    def test_roc_curve(self):
        try:
            y_prob = self.model.predict_proba(self.df)
            self.visualizer.roc_curve(self.model.predict(self.df), y_prob)
        except Exception as e:
            self.fail(f'roc_curve() raised {e} unexpectedly!')

    def test_feature_importance(self):
        try:
            self.visualizer.feature_importance()
        except Exception as e:
            self.fail(f'feature_importance() raised {e} unexpectedly!')

    def test_tsne_plot(self):
        try:
            self.visualizer.tsne_plot()
        except Exception as e:
            self.fail(f'tsne_plot() raised {e} unexpectedly!')

    def test_create_dashboard(self):
        try:
            self.visualizer.create_dashboard(output='test_dashboard.html')
        except Exception as e:
            self.fail(f'create_dashboard() raised {e} unexpectedly!')

if __name__ == '__main__':
    unittest.main()

