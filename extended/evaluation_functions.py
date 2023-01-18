import numpy as np
from sklearn import metrics

def autoencoder_accuracy(probabilities, n_timesteps, n_categories, validation_set, validation_weight, avg_weight):
    pred = np.zeros(probabilities.shape[0]*n_timesteps*n_categories, dtype=np.int8).reshape(probabilities.shape[0], n_timesteps, n_categories)
    for i in range(n_timesteps):
        for j in range(n_categories):
            pred[:,i,j] = np.where(probabilities[:, i, j] == np.amax(probabilities[:, i, :], axis=1),1,0)
    accuracy = np.zeros(n_timesteps)
    for i in range(n_timesteps):
        accuracy[i] = metrics.accuracy_score(validation_set[:,i,:], pred[:,i,:], sample_weight=validation_weight[:,i])
    return np.average(accuracy, weights=avg_weight)

def hit(predictions, test_set, k):
    n_obs = test_set.shape[0]
    rank = (-predictions).argsort()
    top_k_recommendations = rank[:,0:k]
    hit = np.empty([n_obs*k]).reshape(n_obs, k)
    for i in range(n_obs):
        hit[i,:] = test_set[i,top_k_recommendations[i,:]]
    hit = np.max(hit, axis=1)
    return hit

def reciprocal_rank(predictions, test_set, k):
    rank = (-predictions).argsort()
    ranked_items = rank.argsort()
    relevant_items = np.where(test_set == 1, ranked_items, np.nan)
    relevant_items1 = np.where(relevant_items >= k, np.nan, relevant_items)
    min_rank = np.nanmin(relevant_items1, axis=1)
    rr = 1/(min_rank+1)
    rr = np.nan_to_num(rr)
    return rr

def recall(predictions, test_set, k):
    n_obs = test_set.shape[0]
    rank = (-predictions).argsort()
    top_k_recommendations = rank[:,0:k]
    labels = [np.nonzero(t)[0] for t in test_set]
    true_labels_captured = np.empty([n_obs])
    for i in range(n_obs):
        true_labels_captured[i] = len(np.intersect1d(top_k_recommendations[i,:],labels[i]))

    true_labels = np.empty([n_obs])
    for i in range(n_obs):
        true_labels[i] = len(labels[i])

    recall = true_labels_captured/true_labels
    return recall
    
def precision(predictions, test_set, k):
    n_obs = test_set.shape[0]
    rank = (-predictions).argsort()
    top_k_recommendations = rank[:,0:k]
    labels = [np.nonzero(t)[0] for t in test_set]
    true_labels_captured = np.empty([n_obs])
    for i in range(n_obs):
        true_labels_captured[i] = len(np.intersect1d(top_k_recommendations[i,:],labels[i]))

    true_labels = np.empty([n_obs])
    for i in range(n_obs):
        true_labels[i] = len(labels[i])

    precision = true_labels_captured/k
    return precision 

def average_precision(predictions, test_set, k):
    n_obs = test_set.shape[0]
    n_items = test_set.shape[1]
    labels = [np.nonzero(t)[0] for t in test_set]
    precision_at_j = np.empty([n_obs,n_items])
    for j in range(n_items):
        top_j_recommendations = (-predictions).argsort()[:,0:(j+1)]
        
        true_labels_captured = np.empty([n_obs,1])
        for i in range(n_obs):
            true_labels_captured[i] = len(np.intersect1d(top_j_recommendations[i,:],labels[i]))
            
        precision_at_j[:,j:(j+1)] = true_labels_captured/(j+1)
        
    relevant_j = np.empty([n_obs,n_items])
    for j in range(n_items):
        top_j_recommendations = (-predictions).argsort()[:,0:(j+1)]
        for i in range(n_obs):
            relevant_j[i,j] = np.where(len(np.intersect1d(top_j_recommendations[i,j],labels[i]))==1,1,0)

    L = np.empty([n_obs], dtype='int')
    for i in range(n_obs):
        L[i] = min(k,len(labels[i]))

    ap = np.empty([n_obs])
    for i in range(n_obs):
        ap[i] = sum(np.multiply(precision_at_j[:,0:k],relevant_j[:,0:k])[i,:])/L[i]
    return ap