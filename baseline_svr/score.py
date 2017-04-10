from math import floor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# http://www.mt-archive.info/MTMarathon-2012-Specia-ppt.pdf
def deltaAvg(reference, prediction):
    data = [(pred, ref) for pred, ref in zip(prediction, reference)]
    data_sorted = sorted(data, key=lambda x: x[0], reverse=True)
    dataLen = len(data_sorted)
    
    avg = sum([x[1] for x in data_sorted])/dataLen
    deltaLen = floor(dataLen//2+1)
    deltaAvg = [0] * deltaLen
        
    for k in range(2, deltaLen):
        for i in range(1, k):
            deltaAvg[k] += sum([x[1] for x in data_sorted[:floor(dataLen*i/k)]])
        deltaAvg[k] = deltaAvg[k]/(k-1) - avg
    return sum(deltaAvg)/(deltaLen-2)


def printScores(target, prediction):    
    print('Pearson\'s r:', pearsonr(target, prediction)[0])
    print('RMSE:', math.sqrt(mean_squared_error(target, prediction)))
    print('MAE:', mean_absolute_error(target, prediction))
    print('Spearman\'s rank:', spearmanr(target, prediction)[0])
    print('DeltaAvg: ', deltaAvg(target, prediction))
