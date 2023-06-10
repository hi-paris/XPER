**XPER (eXplainable PERformance)** is a methodology designed to measure the specific contribution of the input features to the predictive performance of any econometric or machine learning model. XPER is built on Shapley values and interpretability tools developed in machine learning but with the distinct objective of focusing on model performance (AUC, $R^2$) and not on model predictions ($\hat{y}$). XPER has as a special case the standard explainability method in Machine Learning (SHAP).


![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
## 01 Install 🚀
The library has been tested on Linux, MacOSX and Windows. It relies on the following Python modules:

Pandas
Numpy
Scipy
Scikit-learn

XPER can be installed from [PyPI](https://pypi.org/project/XPER):

<pre>
pip install -i https://test.pypi.org/simple/ XPER
</pre>

#### Post installation check
After a correct installation, you should be able to import the module without errors:

```python
import XPER
```

## 02 XPER example on sampled data step by step ➡️


#### 1️⃣ Load the Data 💽

* Option 1 
```python
import XPER
from XPER.datasets.sample import sample_generation
X_train, y_train, X_test, y_test, p, N, seed  = sample_generation(N=500,p=6,seed=123456)
```
![sample](https://i.postimg.cc/59TwZb8r/Sample.png)


* Option 2
```python

from XPER.datasets.load_data import boston
df = boston()
df.head(3)
```

![boston](https://i.postimg.cc/85TyfDZ4/Boston.png)

#### 2️⃣ Load the trained model or train your model ⚙️

```python
import joblib
model = joblib.load('xgboost_model.joblib')
result = loaded_model.score(X_test, y_test)
print("Model performance: ",result)
```

#### 3️⃣ Monitor Performance 📈

```python
from XPER.models.Performance import evaluate_model_performance
Eval_Metric = ["Precision"]
PM = evaluate_model_performance(Eval_Metric, X_train, y_train, X_test, y_test, model)
print("Performance Metrics: ",PM)
```

![metric](https://i.postimg.cc/Gt5zfDdg/Performance-Metrics.png)

```python
from XPER.models.Performance import calculate_XPER_values
CFP = None
CFN = None
result = calculate_XPER_values(X_test, y_test, model, Eval_Metric, CFP, CFN, PM)
print("Efficiency bench XPER: ", result[-1])
```

## 03 Acknowledgements

The contributors to this library are 
* [Sébastien Saurin](https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=4582330)
* [Christophe Hurlin](https://sites.google.com/view/christophe-hurlin/home)
* [Christophe Pérignon](https://www.hec.edu/fr/faculty-research/faculty-directory/faculty-member/perignon-christophe)



## 04 References

1. *XPER:* Hué, Sullivan, Hurlin, Christophe, Pérignon, Christophe and Saurin Sébastien. "Explainable Performance (XPER): Measuring the Driving Forces of Predictive Performance". HEC Paris Research Paper No. FIN-2022-1463, Available at SSRN: https://ssrn.com/abstract=4280563 or http://dx.doi.org/10.2139/ssrn.4280563, 2022.

