import autoprot.visualization as vis

name = "AKT1S1"
length = 256
domain_position = [(35,43),
                   (77,96)]
ps = [88, 92, 116, 183, 202, 203, 211, 212, 246]
pl = ["pS88", "pS92", "pS116", "pS183", "pS202", "pS203", "pS211", "pS212", "pS246"]
plc = ['C', 'A', 'A', 'C', 'Cd', 'D', 'D', 'B', 'D']
vis.visPs(name, length, domain_position, ps, pl, plc, pls=12)
plt.show()