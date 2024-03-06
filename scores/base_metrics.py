
import numpy as np
import tqdm
from scipy.spatial import ConvexHull

def compute_pvalue_evolution(detector, text,maxsize,split_size):
    pvalues = np.zeros(maxsize)
    for j in tqdm.tqdm((range(split_size, maxsize,split_size))):
        scores_no_aggreg = detector.get_scores_by_t([text], scoring_method='v1', ntoks_max=j)
        #print(scores_no_aggreg[0])
        pvalues[j] = np.array([detector.get_pvalues(scores_no_aggreg[0])])[0]
    return(pvalues)
def quality(data, data_base):
    clean_rating = np.array([(d.id, d.rating) for d in data if d.attack is None and d.watermark is not None])
    baseline_rating = np.array([(d.id, d.rating) for d in data_base if d.watermark is None])[:len(clean_rating )]
    quality = np.zeros(len(clean_rating))
    for id in clean_rating[:,0]:
        #assert id == baseline_rating[int(id),0]
        quality[int(id)] = clean_rating[int(id),1]/baseline_rating[int(id),1]
    #quality = clean_rating[:,1]/baseline_rating[:,1]
    quality = quality[quality !=0]
    return(np.mean(quality))
def compute_size(detector, text,maxsize,split_size, tau=0.02):
    for j in (range(maxsize, 1,-split_size)):
        scores_no_aggreg = detector.get_scores_by_t([text], scoring_method='v1', ntoks_max=j)
        pvalues = np.array([detector.get_pvalues(scores_no_aggreg[0])])[0]
        #print(pvalues, end='\r')
        if pvalues > tau: return(j)
    return(0)
def watermark_size(detector, data,tau=0.02,gen_len=1024,N=32):
    results= [d.response for d in data if d.attack is None and d.watermark is not None]
    E = np.zeros(len(results))
    for i in tqdm.tqdm(range(len(results))):
        E[i] = compute_size(detector, results[i],maxsize=gen_len,split_size=gen_len//N, tau=tau)
    E[E==0] =gen_len//N
    return(np.median(E))
def compute_empirical_power(data,detector,PFA):
    results= [d.response for d in data if d.attack is None and d.watermark is not None]
    pvalues = np.zeros(len(results))
    for i in tqdm.tqdm(range(len(results))):
        scores_no_aggreg = detector.get_scores_by_t([results[i]], scoring_method='v1')
        #print(scores_no_aggreg[0])
        pvalues[i] = np.array([detector.get_pvalues(scores_no_aggreg[0])])[0]
    PD = np.sum((pvalues < PFA))/len(results)
    return(PD)
def tamper_resistance(data, tau=0.02,pval=None,plot=False):
    baseline_rating = np.array([(d.id, d.rating) for d in data if d.attack is None and d.watermark is not None])
    attack_types = np.unique([d.attack for d in data if d.attack is not None and d.watermark is not None])
    ids = np.unique([d.id for d in data if d.attack is not None and d.watermark is not None])

    ATK_ENUM = {}
    for i in range(attack_types.size):
        ATK_ENUM[attack_types[i]] = i

    attack_rating = [(ATK_ENUM[d.attack], d.id, d.rating) for d in data if d.attack is not None and d.watermark is not None]


    QA = np.zeros(attack_types.size+2)
    WA = np.zeros(attack_types.size+2)
    
    
    
    for i in range(attack_types.size):
        if pval is None:
            atk = np.array([(d.id, d.rating, d.pvalue) for d in data if d.attack == attack_types[i] and d.watermark is not None])
        else:
            atk = np.array([(data[j].id, data[j].rating, pval[j]) for j in range(len(data)) 
                            if data[j].attack == attack_types[i] and data[j].watermark is not None])
        for j in ids:
            QA[i] += atk[atk[:,0] == j][0][1]/baseline_rating[j,1]
            if atk[atk[:,0] == j][0][2] < tau: WA[i] += 1
            
    QA = QA/ids.size
    WA = WA/ids.size
    
    QA[-1] =1
    WA[-1] =1

    C = np.vstack([QA,WA]).T
    C[-1,:] = [0,1]

    hull = ConvexHull(C)
    if plot:
        plt.scatter(QA,WA)
        plt.plot(C[hull.vertices,0],C[hull.vertices,1], 'r--', lw=2)
    TR = 2*(1-hull.volume)
    return(TR)