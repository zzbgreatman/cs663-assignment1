class K_Means_Extend(cluster):
    def __init__(self,k=5,max_iter=300,is_balanced=False):
        self.k_ = k;
        self.tolerance_ = 0.0001
        self.max_iter_ = max_iter
        self.is_balanced_ = is_balanced
    def fit(self,X):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = X[i]
        
        self.assignment_={}
        self.bucketsize_ = len(X)/self.k_
                
        for i in range(self.max_iter_):
            self.clusters_={}
            for i in range(self.k_):
                self.clusters_[i] = []
            self.assignment_= []
            for feature in X:
                distances = []
                for center in self.centers_:
                    distances.append(np.linalg.norm(feature-self.centers_[center]))
                if self.is_balanced_:
                    placed = False
                    while(not placed):
                        classification = distances.index(min(distances))
                        if len(self.clusters_[classification]) <= self.bucketsize_:
                            self.clusters_[classification].append(feature)
                            self.assignment_.append(classification)
                            placed = True
                        else:
                            distances.remove(min(distances))
                else:
                    classification = distances.index(min(distances))
                    self.clusters_[classification].append(feature)
                    self.assignment_.append(classification)
            prev_centers = dict(self.centers_)
            for c in self.clusters_:
                self.centers_[c] = np.average(self.clusters_[c],axis=0)
            
            finished = False
            
            for center in self.centers_:
                org_center = prev_centers[center]
                cur_center = self.centers_[center]
                if np.sum((cur_center-org_center)/org_center) > self.tolerance_:
                    finished = True
                
                if finished:
                    break
            
        return self.assignment_, self.centers_