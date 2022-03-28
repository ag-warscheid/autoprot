import autoprot.analysis as ana
import seaborn as sns

x_values = np.random.randint(-50,110,size=(250))
y_values = np.square(x_values)/1.5 + np.random.randint(-1000,1000, size=len(x_values))
df = pd.DataFrame({"Xvalue" : x_values,
                   "Yvalue" : y_values
                   })
evalDF = ana.loess(df, "Xvalue", "Yvalue", alpha=0.7, poly_degree=2)
fig, ax = plt.subplots(1,1)
sns.scatterplot(df["Xvalue"], df["Yvalue"], ax=ax)
ax.plot(evalDF['v'], evalDF['g'], color='red', linewidth= 3, label="Test")
plt.show()