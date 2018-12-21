'''
Created on Oct 18, 2018

@author: Jonathan Schertz
'''

import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates as pc
from sklearn import cluster, ensemble, multiclass, metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import itertools
from scipy.spatial.distance import cdist
import os

NEEDS = list(range(12))
BIG_5 = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Emotional range"]
VALUES = list(range(47,52))
ROT = -50

# Setup cluster names
NAMING_FILE = os.path.join(os.path.dirname(__file__),"big5_names.txt")
NAMING = pd.read_csv(NAMING_FILE, sep='\t', index_col=0)

def test():
    a = Viz('data/Fast50/Fast50.csv')
    a.filter('big5')
    a.clusterKmeans()
    return a

class Viz:
    
    def __init__(self, file):
        self.df_base = pd.read_csv(file, index_col = 0)
        self.df_filter = None
        self.df = self.df_base
        self.cluster = None
        self.importances = None
        self.loc = os.path.dirname(file) + "/"
        self.name = os.path.basename(file)[:os.path.basename(file).find('.')]
        
    def plotParallel(self, name = None):
        if self.cluster is None:
            temp_df = self.df.assign(cluster=['no cluster'] * self.df.shape[0])
        else:
            temp_df = self.df.assign(cluster=list(self.cluster))
        plt.figure(figsize=(16,10))
        pc(temp_df, class_column = 'cluster', colormap = "plasma")
        plt.xticks(rotation=ROT)
        plt.tight_layout(1)
        if name is None:
            plt.show()
        elif type(name) is PdfPages:
            name.savefig()
            plt.close()
        else:
            plt.savefig(self.loc + name + '.png')
            plt.close()
        
    def plotBox(self, m=2, name=None):
        if self.cluster is None:
            self.df.boxplot(grid=False)
            plt.show()
        else:
            n = np.unique(self.cluster)
            r = math.ceil(math.sqrt(len(n)))
            c = math.ceil(len(n)/r)
            axes = enumerate(itertools.chain(*(plt.subplots(nrows=r, ncols=c, figsize=(16,10))[1])))
            for clu in n:
                frame = self.df[self.cluster == clu]
                imps = self.importances[clu]
                maxes = [idx[0] for idx in sorted(enumerate(imps), key = lambda x : x[1], reverse = True)]
                plt.sca(next(axes)[1])
                plt.title("Cluster " + str(clu) + " (n=" + str(frame.shape[0]) + ")")
                plt.ylabel("Percentile")
                plt.ylim(0,1)
                bp = frame.boxplot(grid=False, rot=ROT, return_type='dict')
                for i in range(m):
                    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                        if len(bp[element]) > len(imps):
                            es = [bp[element][maxes[i]*2],bp[element][maxes[i]*2+1]]
                        else:
                            es = [bp[element][maxes[i]]]
                        [plt.setp(e, color='red') for e in es]
            if name is None:
                plt.tight_layout(0, -3, -3)
                plt.show()
            elif type(name) is PdfPages:
                plt.tight_layout()
                name.savefig()
                plt.close()
            else:
                plt.tight_layout()
                plt.savefig(self.loc + name + '.png')
                plt.close()
            
    def plotImportances(self, m=2, name=None):
        if self.importances is None:
            print("Importances have not yet been calculated")
            return
        n = np.unique(self.cluster)
        r = math.ceil(math.sqrt(len(n)))
        c = math.ceil(len(n)/r)
        axes = enumerate(itertools.chain(*(plt.subplots(nrows=r, ncols=c, figsize=(16,10))[1])))
        for clu in n:
            
            # Get subset and importances
            frame = self.df[self.cluster == clu]
            imps = self.importances[clu].copy()
            maxes = [idx[0] for idx in sorted(enumerate(imps), key = lambda x : x[1], reverse = True)]

            # Setup plot
            plt.sca(next(axes)[1])
            x_pos = np.arange(len(imps))
            plt.title("Cluster " + str(clu) + " (n=" + str(frame.shape[0]) + ")")
            plt.ylim(-.8,.8)
            plt.ylabel("Feature Importance")
            plt.xticks(x_pos, list(self.df), rotation=ROT)
            plt.axhline(y=0, color='k')
            
            # Set colors, direction, and text
            edgecolors = ['blue'] * len(imps)
            for i in range(len(imps)):
                feature = list(self.df)[i]
                mean = frame.mean()[feature]
                mean_other = self.df[self.cluster != clu].mean()[feature]
                mean_diff = mean - mean_other
                offset = .05
                if mean_diff < 0:
                    imps[i] = -imps[i]
                    offset = -.15
                if i in maxes[:m]:
                    edgecolors[i] = 'red'
                    plt.text(x_pos[i] -.1, imps[i] + offset, "{0:.2f}".format(mean - mean_other))

            # Plot bars
            plt.bar(x_pos, imps, edgecolor=edgecolors)

        if name is None:
            plt.tight_layout(0, -3, -3)
            plt.show()
        elif type(name) is PdfPages:
            plt.tight_layout()
            name.savefig()
            plt.close()
        else:
            plt.tight_layout()
            plt.savefig(self.loc + name + '.png')
            plt.close()
        
    def showImportances(self, m=2):
        if self.importances is None:
            print("Importances have not yet been calculated")
            return
        for clu in range(len(self.importances)):
            imps = self.importances[clu]
            maxes = [idx[0] for idx in sorted(enumerate(imps), key = lambda x : x[1], reverse = True)]
            print("\nCluster " + str(clu) + ":")
            for i in range(m):
                idx = maxes[i]
                feature = list(self.df)[idx]
                mean = self.df[self.cluster == clu].mean()[feature]
                mean_other = self.df[self.cluster != clu].mean()[feature]
                mean_diff = mean - mean_other
                hl = "\tHigh " if mean_diff > 0 else "\tLow "
                print(hl + feature + " (" + "{0:.2f}".format(mean_diff) + ")")
            
    def calcImportances(self):
        if self.cluster is None:
            print("Data must be clustered before importances can be calculated")
            return
        estimator = ensemble.RandomForestClassifier(n_estimators=100)
        classifier = multiclass.OneVsRestClassifier(estimator)
        classifier.fit(self.df, self.cluster)
        self.importances = [e.feature_importances_ for e in classifier.estimators_]
        
    def clusterKmeans(self, n = 4):
        self.cluster = cluster.KMeans(n_clusters = n).fit(self.df).labels_
        self.calcImportances()
        
    def clusterAffinity(self):
        self.cluster = cluster.AffinityPropagation().fit(self.df).labels_
        self.calcImportances()
        
    def clusterAgglomerative(self, n = 4):
        self.cluster = cluster.AgglomerativeClustering(n_clusters=n).fit(self.df).labels_
        self.calcImportances()
        
    def kmeansElbow(self):
        distortions = []
        inertias = []
        for k in range(1, 11):
            model = cluster.KMeans(n_clusters = k).fit(self.df)
            distortions.append(sum(np.min(cdist(self.df, model.cluster_centers_, 'euclidean'), axis=1)) / self.df.shape[0])
            inertias.append(model.inertia_)
        ax = plt.subplot(1,2,1)
        ax.plot(range(1, 11), distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        ax = plt.subplot(1,2,2)
        ax.plot(range(1, 11), inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method showing the optimal k (inertia)')
        plt.show()
    
    def printClusters(self, name = "clusters"):
        if self.cluster is None:
            print("Cluster not yet defined")
            return
        
        # Get users in each cluster
        frames = []
        names = []
        for n in np.unique(self.cluster):
            f = pd.DataFrame(self.df[self.cluster == n].index)
            imps = self.importances[n]
            maxes = [idx[0] for idx in sorted(enumerate(imps), key = lambda x : x[1], reverse = True)]
            features = []
            for i in range(2):
                idx = maxes[i]
                feature = list(self.df)[idx]
                mean = self.df[self.cluster == n].mean()[feature]
                mean_other = self.df[self.cluster != n].mean()[feature]
                mean_diff = mean - mean_other
                hl = "-High" if mean_diff > 0 else "-Low"
                features.append(feature + hl)
            frames.append(f)
            names.append(NAMING.loc[features[0],features[1]])
            
        # Combine clusters into single dataframe
        users = pd.concat(frames, axis=1, ignore_index=True)
        users.columns = names
        users.fillna("", inplace=True)
        
        # Output
        if type(name) is PdfPages:
            fig, ax = plt.subplots()
            fig.patch.set_visible(False)
            fig.suptitle('Users by Cluster')
            ax.axis('off')
            ax.axis('tight')
            plt.table(cellText = users.values, colLabels = users.columns, loc = 'center')
            fig.tight_layout()
            name.savefig()
        else:
            users.to_csv(self.loc + name + ".csv", index=False)

    def compareCluster(self, f1 = 'big5', c1 = 'kmean', f2 = 'big5', c2 = 'agg'):
        clusters = {
            'kmean' : Viz.clusterKmeans,
            'agg' : Viz.clusterAgglomerative,
            'aff' : Viz.clusterAffinity }
        c1 = clusters[c1]
        c2 = clusters[c2]

        # Get first cluster sets
        self.filter(f1)
        c1(self)
        d1 = {}
        for clu in list(set(self.cluster)):
            d1[clu] = set(self.df[self.cluster == clu].index)
            
        # Get second cluster sets
        self.filter(f2)
        c2(self)
        d2 = {}
        for clu in list(set(self.cluster)):
            d2[clu] = set(self.df[self.cluster == clu].index)
            
        # Calculate ratios and plot
        ax = plt.subplot(111)
        x_pos = np.arange(len(d2))
        w = .2
        for k1 in d1.keys():
            r = []
            for k2 in d2.keys():
                i = len(d1[k1].intersection(d2[k2]))
                if i == 0:
                    r.append(.01)
                else:
                    u = len(d1[k1].union(d2[k2]))
                    r.append(i / float(u))
            ax.bar(x_pos + w * k1, r, w, label = k1)
        plt.legend(title = c1.__name__ + "(" + f1 + ")")
        plt.xticks(x_pos + w * len(d2) / 2, d2.keys())
        plt.xlabel(c2.__name__ + "(" + f2 + ")")
        plt.ylabel('Similarity')
        plt.title('Comparing Clusters')
        plt.show()
        
    def assessCluster(self, c = 'kmean', **kwargs):
        clusters = {
            'kmean' : cluster.KMeans,
            'agg' : cluster.AgglomerativeClustering,
            'aff' : cluster.AffinityPropagation }
        c = clusters[c]
        print(kwargs)
        base = c(**kwargs).fit(self.df).labels_
        accuracies = [[],[],[]]
        for _ in range(10):
            msk = np.random.rand(len(self.df)) < .8
            train = self.df[msk]
            test = self.df[~msk]
            trained = c(**kwargs).fit(train)

            # Check for cluster flips
            flips = {}
            for j in np.unique(base):
                unique, counts = np.unique(trained.labels_[base[msk] == j], return_counts = True)
                if(len(unique)==0): continue
                max_idx = np.argmax(counts)
                if unique[max_idx] != j:
                    flips[j] = unique[max_idx]
                print("Cluster " + str(j) + ": " + "{0:.0%}".format(counts[max_idx]/float(sum(counts))))
            act = trained.predict(test)
            exp = base[~msk]
            exp = list(map(lambda x : flips.get(x, x), exp))
            
            accuracy = np.sum(exp == act)/float(len(act))
            precision, recall, _, _ = metrics.precision_recall_fscore_support(act, exp, average='macro')
            accuracies[0].append(accuracy)
            accuracies[1].append(precision)
            accuracies[2].append(recall)
            print("Accuracy: " + "{0:.0%}".format(accuracy))
            print("Precision: " + "{0:.0%}".format(precision))
            print("Recall: " + "{0:.0%}".format(recall))
            
        print("Average Accuracy: " + "{0:.0%}".format(sum(accuracies[0])/float(len(accuracies[0]))))
        print("Average Precision: " + "{0:.0%}".format(sum(accuracies[1])/float(len(accuracies[1]))))
        print("Average Recall: " + "{0:.0%}".format(sum(accuracies[2])/float(len(accuracies[2]))))

    def filter(self, f, retain_clusters=False):
        filters = {
            "big5" : (self.filterItems,BIG_5),
            "opn" : (self.filterLike,BIG_5[0] + "."),
            "con" : (self.filterLike,BIG_5[1] + "."),
            "ext" : (self.filterLike,BIG_5[2] + "."),
            "agr" : (self.filterLike,BIG_5[3] + "."),
            "emo" : (self.filterLike,BIG_5[4] + "."),
            "need" : (self.filterRange,NEEDS),
            "val" : (self.filterRange,VALUES) }
        filt = filters.get(f, (print, "Invalid filter"))
        filt[0](filt[1])
        self.filterOn(retain_clusters)
    
    def filterOn(self, retain_clusters):
        if self.df_filter is None:
            print("Error: no filter set")
        else:
            self.df = self.df_filter
            if not retain_clusters:
                self.cluster = None
                self.importances = None
        
    def filterOff(self):
        self.df = self.df_base
                
    def filterItems(self, items):
        self.df_filter = self.df_base.filter(items=items)
        
    def filterLike(self, like):
        self.df_filter = self.df_base.filter(like=like)
    
    def filterRange(self, idx):
        self.df_filter = self.df_base.iloc[:,idx]
        
    def report(self, name = None):
        name = self.name if name is None else name
        print('[' + name + '] Generating report...')
        
        # Filter and cluster by big5
        self.filter('big5')
        self.clusterKmeans()
        
        # Collect feature importances of the clusters
        clusters = []
        names = []
        for clu in range(len(self.importances)):
            cluster_features = [name, self.df[self.cluster == clu].shape[0]]
            imps = self.importances[clu]
            maxes = [idx[0] for idx in sorted(enumerate(imps), key = lambda x : x[1], reverse = True)]
            for i in range(2):
                idx = maxes[i]
                feature = list(self.df)[idx]
                mean = self.df[self.cluster == clu].mean()[feature]
                mean_other = self.df[self.cluster != clu].mean()[feature]
                mean_diff = mean - mean_other
                hl = "-High" if mean_diff > 0 else "-Low"
                cluster_features.append(feature + hl)
            clusters.append(cluster_features) 
            names.append(NAMING.loc[cluster_features[2],cluster_features[3]])
        self.printClusters()

        # Setup first page of report
        cluster_desc = {}
        cluster_desc['Id'] = ['Cluster ' + str(x) for x in range(len(clusters))]
        cluster_desc['Size'] = ['n=' + str(x) for x in list(zip(*clusters))[1]]
        cluster_desc['Description'] = names
        cluster_desc['Tweet/Follower'] = [(self.df_base['Tweets']/self.df_base['Followers'])[self.cluster==c].mean() for c in range(len(clusters))]
        cluster_desc['Follower/Following'] = [(self.df_base['Followers']/self.df_base['Following'])[self.cluster==c].mean() for c in range(len(clusters))]
        cluster_desc_frame = pd.DataFrame().assign(**cluster_desc)[['Id','Size','Description','Tweet/Follower','Follower/Following']]
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        fig.suptitle('Cluster Descriptions')
        ax.axis('off')
        ax.axis('tight')
        plt.table(cellText = cluster_desc_frame.values, cellLoc = 'left', colLabels = cluster_desc_frame.columns, loc = 'center')
        fig.tight_layout()
        
        for s in ['big5', 'need', 'val']:
            self.filter(s, retain_clusters=True)
            self.calcImportances()
            pp = PdfPages(self.loc + name + '_' + s + ".pdf")
            pp.savefig(fig)
            self.plotParallel(name=pp)
            self.plotBox(name=pp)
            self.plotImportances(name=pp)
            pp.close()
            
        # Return clusters
        return clusters
    