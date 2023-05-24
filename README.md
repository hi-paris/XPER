**XPER (eXplainable PERformance)** is a methodology designed to measure the specific contribution of the input features to the predictive performance of any econometric or machine learning model. XPER is built on Shapley values from game theory and interpretability tools developed in machine learning but with the distinct objective of focusing on model performance (AUC, $R^2$) and not on model predictions ($\hat{y}$). XPER has as a special case the standard explainability method in Machine Learning (SHAP).

## Install

XPER can be installed from [PyPI](https://pypi.org/project/XPER):

<pre>
pip install XPER
</pre>

## XGBoost example

```python
import xgboost
import XPER

# train an XGBoost model
X, y = XPER.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

# Measure the impact of each feature on the performance of the model using XPER
```

## References

1. *XPER:* Hué, Sullivan, Hurlin, Christophe, Pérignon, Christophe and Saurin Sébastien. "Explainable Performance (XPER): Measuring the Driving Forces of Predictive Performance". HEC Paris Research Paper No. FIN-2022-1463, Available at SSRN: https://ssrn.com/abstract=4280563 or http://dx.doi.org/10.2139/ssrn.4280563, 2022.

