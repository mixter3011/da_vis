# examples/example_usage.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from da_vis.visualizer import DataVisualizer 

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

model = RandomForestClassifier()
model.fit(df, data.target)

y_prob = model.predict_proba(df)

visualizer = DataVisualizer(data=df, model=model)

visualizer.correlation_heatmap()
visualizer.feature_distribution('sepal length (cm)')
visualizer.confusion_matrix(data.target, model.predict(df))
visualizer.roc_curve(data.target, y_prob)
visualizer.feature_importance()
visualizer.tsne_plot()

visualizer.create_dashboard()
