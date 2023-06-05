# ##### Summary plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ordering = shap_values.mean(0)              # By default the features are ordered
                                            # using shap_values.abs.mean(0). We
                                            # change it to the mean value of the
                                            # XPER values = phi_j (global ones)

shap.plots.beeswarm(shap_values, order=ordering,show=False)
plt.xlabel("Contribution")
#plt.savefig('./Figures/Summary_plots.pdf', format='pdf', dpi = 1200, bbox_inches='tight')
plt.show()