import seaborn as sns
import autoprot.clustering as clst

df = sns.load_dataset('iris')
labels = df.pop('species')
c = clst.KMeans(df)
c.auto_run()