import autoprot.analysis as ana

df = pd.DataFrame({"a1":np.random.normal(loc=0, size=4000),
                   "a2":np.random.normal(loc=0, size=4000),
                   "a3":np.random.normal(loc=0, size=4000),
                   "b1":np.random.normal(loc=0.5, size=4000),
                   "b2":np.random.normal(loc=0.5, size=4000),
                   "b3":np.random.normal(loc=0.5, size=4000),})
testRes = ana.limma(df, reps=[["a1","a2", "a3"],["b1","b2", "b3"]], cond="_test")
testRes["P.Value_test"].hist()
plt.show()