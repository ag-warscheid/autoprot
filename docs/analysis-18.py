import autoprot.analysis as ana
import pandas as pd
df = pd.DataFrame({"a1":np.random.normal(loc=0, size=4000),
          "a2":np.random.normal(loc=0, size=4000),
          "a3":np.random.normal(loc=0, size=4000),
          "b1":np.random.normal(loc=0.5, size=4000),
          "b2":np.random.normal(loc=0.5, size=4000),
          "b3":np.random.normal(loc=0.5, size=4000),})
ana.ttest(df=df,
          reps=[["a1","a2", "a3"],["b1","b2", "b3"]])["pValue_"].hist(bins=50)
plt.show()