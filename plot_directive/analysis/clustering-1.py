df = sns.load_dataset('iris')
labels = df.pop('species')
c = ana.KMeans(df)
c.auto_run()