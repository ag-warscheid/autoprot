import pandas as pd
import autoprot.visualization as vis

arrays = [(False,False,False,False,False,False,True,False),
          (False,False,False,False,False,False,False,True),
          (False,False,False,False,False,True,False,False),
          (False,False,True,False,False,False,False,False),
          (False,False,False,False,True,False,False,False),
          (False,False,False,True,False,False,False,False),
          (True,False,False,False,False,False,False,False),
          (False,True,False,False,False,False,False,False),
          (False,False,False,False,False,False,True,True),
          (False,False,False,False,True,True,False,False),
          (False,False,False,False,False,True,True,False),
          (True,True,False,False,False,False,False,False),
          (False,True,True,False,False,False,False,False),
          (True,False,True,False,False,False,False,False),
          (False,False,True,True,False,False,False,False),
          (False,False,False,False,False,True,False,True),
          (False,False,False,False,True,False,True,False),
          (False,False,False,False,True,False,False,True),
          (False,True,False,True,False,False,False,False),
          (False,False,True,False,True,False,False,False),
          (False,False,False,True,True,False,False,False),
          (True,False,False,False,False,False,False,True),
          (False,False,False,False,True,True,True,False),
          (False,False,False,False,False,True,True,True),
          (False,False,False,False,True,False,True,True),
          (True,True,True,False,False,False,False,False),
          (False,True,True,True,False,False,False,False),
          (False,False,False,False,True,True,True,True),
          (True,True,True,True,False,False,False,False)]
arrays = np.array(arrays).T
values = (106,85,50,30,29,26,17,14,94,33,30,19,13,9,7,5,4,2,1,1,1,1,102,29,14,11,1,60,3)
example = pd.Series(values,
                    index=pd.MultiIndex.from_arrays(arrays,
                                                    names=('120_down',
                                                           '60_down',
                                                           '30_down',
                                                           '10_down',
                                                           '120_up',
                                                           '60_up',
                                                           '30_up',
                                                           '10_up')
                                                   )
                   )

upset = vis.UpSetGrouped(example,
                         show_counts=True,
                         #show_percentages=True,
                         sort_by=None,
                         sort_categories_by='cardinality',
                         facecolor="gray")
upset.styling_helper('up', facecolor='darkgreen', label='up regulated')
upset.styling_helper('down', facecolor='darkblue', label='down regulated')
upset.styling_helper(['up', 'down'], facecolor='darkred', label='reversibly regulated')
specs = upset.plot()
upset.replot_totals(specs=specs, color=['darkgreen',
                                        'darkgreen',
                                        'darkgreen',
                                        'darkgreen',
                                        'darkblue',
                                        'darkblue',
                                        'darkblue',
                                        'darkblue',])

plt.show()