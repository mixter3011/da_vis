# da_vis

`da_vis` is a Python package for visualizing data and machine learning model performance. It provides tools for generating various types of plots and dashboards to analyze and present data insights.

## Installation

You can install `da_vis` using pip:

```bash
pip install da-vis
```
## Usage

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from da_vis.visualizer import DataVisualizer

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

model = RandomForestClassifier()
model.fit(df, data.target)

visualizer = DataVisualizer(data=df, model=model)

visualizer.correlation_heatmap()
visualizer.feature_distribution('sepal length (cm)')
visualizer.confusion_matrix(data.target, model.predict(df))
visualizer.roc_curve(data.target, model.predict_proba(df))
visualizer.feature_importance()
visualizer.tsne_plot()
```
## Features

- **Correlation Heatmap**: Visualizes the correlation matrix of a dataset.
- **Feature Distribution**: Plots the distribution of a specific feature.
- **Confusion Matrix**: Displays the confusion matrix for model evaluation.
- **ROC Curve**: Generates the ROC curve for binary classification models.
- **Feature Importance**: Shows feature importance scores for the model.
- **t-SNE Plot**: Creates a t-SNE plot for visualizing high-dimensional data.

## Contributing

Contributions are welcome! Feel free to submit bug reports, feature requests, or pull requests through GitHub issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
