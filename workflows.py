from autoprot import preprocessing as pp
from autoprot import visualization as vis
from autoprot import analysis as ana
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime



class workflow():
    def __init__(self, file, saveDir, replicates=[], intCols="", 
                corrAnalysis='none', #'none', 'small', 'complete'
                missAnalysis='none', #'none', 'text', 'vis', 'complete'
                minValid = 1,
                test = "limma", #"limma", "standard"
                 ):
        self.data = pp.read_csv(file)
        self.saveDir = saveDir
        self.replicates=replicates
        self.intCols=intCols
        self.corrAnalysis = corrAnalysis
        self.missAnalysis = missAnalysis
        self.minValid = minValid
        self.test = test
        self.replicateNames = [f"Rep{i+1}" for i in range(len(self.replicates))]
        self.ratioCols = np.ravel(self.replicates)
        self.pt = 0.05 #pvalue threshold
        self._makeDir()
    
    
    def _makeDir(self):
        if "autoProt_workflow" in os.listdir(self.saveDir):
            pass
        else:
            os.mkdir(self.saveDir + "/autoProt_workflow")
        self.saveDir = self.saveDir + "/autoProt_workflow"
        today = datetime.datetime.today().strftime("%Y%m%d")
        if today in os.listdir(self.saveDir):
            pass
        else:
            os.mkdir(self.saveDir + "/" + today)
        self.saveDir = self.saveDir + "/" + today
        extraDirs = ["figures", "data"]
        for edir in extraDirs:
            if edir in os.listdir(self.saveDir):
                pass
            else:
                os.mkdir(self.saveDir +"/" + edir)
    
    
    def getRatioCols(self, normalized=True, re=None):
        if re:
            return self.data.filter(regex=re).columns.to_list()
        else:
            if normalized == True:
                return self.data.filter(regex="Ratio ./. normalized.*").columns.to_list()
            else:
                return self.data.filter(regex="Ratio ./. (?!normalized).*").columns.to_list()
                
                
    def getIntensityCols(self, re=None):
        if re:
            return self.data.filter(regex=re).columns.to_list()
        else:
            return self.data.filter(regex="Intensity . .*").columns.to_list()
                
                
    # def setRatioCols(self, ratios):
        # self.ratioCols = ratios
        
        
    def setReplicates(self, replicates, names=None, log=None, invert=None):
        """
        :@log: int, 2 or 10 - base of log
        """
        self.replicates = replicates
        self.ratioCols = np.ravel(self.replicates)
        if names:
            self.replicateNames = names
        else:
            self.replicateNames = [f"Rep{i+1}" for i in range(len(self.replicates))]
        if log:
            for i in range(len(self.replicates)):
                if invert:
                    self.data, self.replicates[i] = pp.log(self.data, self.replicates[i], base=log,
                                                            invert=invert[i], returnCols=True)
                else:
                    self.data, self.replicates[i] = pp.log(self.data, self.replicates[i], base=log, returnCols=True)
        
        
    def setIntensityCols(self, intensities, log=False):
        self.intCols = intensities
        if log==True:
            self.data, self.intCols = pp.log(self.data, self.intCols, returnCols=True)
            
            
    def getSig(self, name, which="both", pt=None):
        """
        returns dataframe with significant entries
        :@which: both, up, down; return all only up or down regulated entries
        :@pt: sets a pt different than the overall pt
        """
        if not pt:
            pt = self.pt
        if which == "both":
            return self.data[self.data[f"adj.P.Val_{name}"]<self.pt]
        elif which == "up":
            return self.data[(self.data[f"logFC_{name}"]>0) & 
                       (self.data[f"adj.P.Val_{name}"]<self.pt)]
        elif which == "down":
            return self.data[(self.data[f"logFC_{name}"]<0) & 
                       (self.data[f"adj.P.Val_{name}"]<self.pt)]
            
            
    def filterSig(self, name):
        up = self.getSig(name, "up")
        down = self.getSig(name, "down")
        print(f"{up.shape[0]} proteins are significantly up-regulated at a p threshold of {self.pt}")
        print(f"{down.shape[0]} proteins are significantly down-regulated at a p threshold of {self.pt}")
        
        if self.saveDir:
            pp.to_csv(up, self.saveDir+f"/data/{name}_sigUpData.tsv")
            with open(self.saveDir+f"/data/{name}_sigUpGenenames.txt", 'w') as f:
                f.write(f"{name} - upregulated")
                f.write('\n')
                for i in up["Gene names"].fillna("").apply(lambda x: x.split(';')[0]):
                    f.write(i)
                    f.write('\n')
            pp.to_csv(down, self.saveDir+f"/data/{name}_sigDownData.tsv")
            with open(self.saveDir+f"/data/{name}_sigDownGenenames.txt", 'w') as f:
                f.write(f"{name} - downregulated")
                f.write('\n')
                for i in down["Gene names"].fillna("").apply(lambda x: x.split(';')[0]):
                    f.write(i)
                    f.write('\n')
        
                       
                       
    def missAna(self):
        if self.missAnalysis == 'none':
            return True
        else:
            print("-"*50)
            print("Miss analysis:")
            if self.missAnalysis == 'text':
                ana.missAnalysis(self.data, self.ratioCols, text=True, vis=False, extraVis=False,
                                saveDir=self.saveDir+"/figures")
                plt.show()
            elif self.missAnalysis == 'vis':
                ana.missAnalysis(self.data, self.ratioCols, text=False, vis=True, extraVis=True,
                                saveDir=self.saveDir+"/figures")
                plt.show()
            elif self.missAnalysis == 'complete':
                ana.missAnalysis(self.data, self.ratioCols, text=True, vis=True, extraVis=True,
                                saveDir=self.saveDir+"/figures")
                plt.show()
            else:
                print('This is not a valid paramter for missAnalysis.')
            
            
    def corrAna(self):
        if self.corrAnalysis == 'none':
            return True
        else:
            print("-"*50)
            print("Correlation analysis:")
            if self.corrAnalysis == 'small':
                vis.corrMap(self.data, self.intCols, cluster=True, annot=True)
                if self.saveDir:
                    plt.savefig(self.saveDir + "/figures/corrMap.pdf")
                plt.show()
            elif self.corrAnalysis == 'complete':
                vis.corrMap(self.data, self.intCols, cluster=True, annot=True)
                if self.saveDir:
                    plt.savefig(self.saveDir + "/figures/corrMap.pdf")
                plt.show()
                vis.correlogram(self.data, self.intCols, file=self.ratioCols,log=True,saveDir = self.saveDir+"/figures", 
                                saveType="pdf", saveName="correlogram", lowerTriang="hist2d",
                sampleFrac=None, bins=100)
                plt.show()
            else:
                print('This is not a valid paramter for corrAnalysis.')
                
                


class proteomeLabeled(workflow):
    def __init__(self, data, saveDir, 
                corrAnalysis='none', #'none', 'small', 'complete'
                missAnalysis='none', #'none', 'text', 'vis', 'complete'
                minValid=1, 
                test="limma", #"limma", "standard"
                intCols="",
                ratios="",
                replicates=[], filetype="proteinGroups"):
                
        #super(proteomeLabeled, self).__init__(data, saveDir, intCols, replicates)
        super().__init__(data, saveDir, replicates, intCols,corrAnalysis, missAnalysis,
        minValid,test)
        self.filetype = filetype


    def DEAna(self):
        for rep, name in zip(self.replicates, self.replicateNames):
                    print(f"Analysis of {name}")
                    if self.test == "limma":
                        self.data = ana.limma(self.data, rep, cond=f"_{name}")
                    elif self.test == "standard":
                        self.data = ana.ttest(self.data, rep, cond=f"_{name}")
                    print("-"*50)
                    vis.volcano(self.data, logFC=f"logFC_{name}", p=f"adj.P.Val_{name}",
                                pt=self.pt, title=name)
                    if self.saveDir:
                        plt.savefig(self.saveDir + f"/figures/{name}_volcano.pdf")
                    plt.show()
                    self.filterSig(name)
                    print("-"*50)


    def preprocessing(self):
        print("PREPROCESSING")
        print("-"*50)
        print("Data cleaning:")
        self.data = pp.cleaning(self.data, self.filetype)
        print("-"*50)
        print("Entries without quantitative data are removed:")
        self.data = pp.removeNonQuant(self.data, self.ratioCols)
        self.missAna()
        self.corrAna()
        print("-"*50)
        print(f"Filter for {self.minValid} valid values / replicate:")
        self.data = pp.filterVv(self.data, self.replicates, n=self.minValid)
        print("-"*50)
        print("-"*50)
                            
                            
    def analysis(self):
        print("ANALYSIS")
        print("-"*50)
        self.DEAna()
        pp.to_csv(self.data, self.saveDir+"/data/analyzedData.tsv")