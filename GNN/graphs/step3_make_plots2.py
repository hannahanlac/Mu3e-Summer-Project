from os import listdir
from pyexpat import model
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplhep as hep
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, log_loss
from sklearn.metrics import confusion_matrix
import seaborn as sns

def prepare_test_data(data):
    f=open(data, 'rb')
    data = pickle.load(f)  
    X = tf.expand_dims(data["events"], -1)
    y = tf.expand_dims(data["labels"], -1)

    return X, y

def eff_rate(fpr, tpr,tn, fp, fn, tp, bg_rate):    
    rates = fpr*bg_rate   
    rates_errs = np.zeros(rates.shape)
    effs = tpr
    effs_errs = np.zeros(effs.shape)
    effs_errs = np.sqrt((effs * (1-effs)) / (tp+fn))
    rates_errs = np.sqrt((fpr * (1-fpr)) / (fp+tn))*bg_rate
    #rates = np.divide(rates, 1000)     #not needed as it is already in kHz??
    #rates_errs = np.divide(rates_errs, 1000)

    return rates, rates_errs, effs, effs_errs

def accuracy_loss(metrics, path, model):
    plt.figure(figsize=(8,5), dpi=200)
    hep.style.use("CMS")
    f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    ax1.plot(metrics['accuracy'])
    ax1.plot(metrics['val_accuracy'])
    #ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='lower left')
    ax2.plot(metrics['loss'])
    ax2.plot(metrics['val_loss'])
    #ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'validation'], loc='upper left')
    plt.title(f"{model}",loc='center')   
    plt.show()
    plt.savefig(f"{path}/training_curves_{model}.png")

def confusion_matrix_plotter(scores, y_test, outdir, model, min_tr=0.5):
    matrix_confusion = confusion_matrix(y_test, scores)
    tn, fp, fn, tp = confusion_matrix(y_test, scores).ravel()
    plt.figure(figsize=(10,8), dpi=200)
    sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.ylabel("y true")
    plt.xlabel("y pred")
    plt.title(f"{model}: {min_tr}",loc='center')   
    plt.savefig(f'{outdir}/confusion_matrix_{min_tr}.png')
    return tn, fp, fn, tp

def get_accuracy(tn, fp, fn, tp):
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    return accuracy

def bkg_rates(scores,threshold):
    "false postive = number of events surviving threshold"
    "True negative = number of events killed by threshold"
    false_count = scores[scores>threshold]
    fp = false_count.shape[0]
    all_count = scores.shape[0]
    fpr = fp / all_count
    return (fpr) *  (40*10**3) * (2760 / 3564) #return rate in kHz

def sig_rates(scores,threshold):
    "True positive = number of events surviving threshold"
    "false negative = number of events killed by threshold"
    false_count = scores[scores<threshold]
    fn = false_count.shape[0]
    all_count = scores.shape[0]
    fnr = fn / all_count
    return (fnr) *  (40*10**3) * (2760 / 3564)

def rate_threshold_curve(rates, thresholds, outdir, model):  
    a = np.array(rates) - 10
    b = a[a<=0]
    idx = a.tolist().index(b[0])
    min_thr = thresholds[idx]

    plt.figure(figsize=(8,6), dpi=200) 
    hep.style.use("CMS")
    ax = plt.axes()
    plt.plot(thresholds, rates, label="Background survival rate", color = "orange")
    plt.yscale('log')
    plt.axhline(y=10, color='black', linestyle='--', label="10KHz rate")
    plt.xlabel("Threshold A.U")
    plt.ylabel("Rate KHz")
    plt.legend()
    plt.title(f"{model}",loc='center')   
    ax.set_facecolor('white')
    plt.savefig(f"{outdir}/bkg_rates.png", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None, 
        )
    return min_thr

def calc_volatility(metrics):
    mses = 0
    for i in range(len(metrics['loss'])):
      mses += (metrics['loss'][i] - metrics['val_loss'][i])**2

    stdev = np.sqrt(mses / (len(metrics['loss']) - 1))

    vol = 0
    for i in range(len(metrics['val_loss']) - 1):
            vol += (metrics['val_loss'][i+1] - metrics['val_loss'][i] - (metrics['loss'][i+1] - metrics['loss'][i]))**2
    vol /= len(metrics['val_loss'])/100

    return stdev, vol

def optimal_eff_rate(effs, rates, effs_errs=None, rates_errs=None):
    
    if effs.shape != rates.shape:
        raise ValueError(f"Shapes {effs.shape} and {rates.shape} are not aligned.")


    uniq_rates = set()
    opt_effs = []
    
    # store indices of optimal efficiencies
    # for indexing pre-computed errors
    idx = []
    
    for _idx, i in enumerate(effs):
        if not rates[_idx] in uniq_rates:
            uniq_rates.add(rates[_idx])
            opt_effs.append(i)
            idx.append(_idx)
        elif i > opt_effs[-1]:
            opt_effs[-1] = i
            idx[-1] = _idx
        
    if effs_errs is not None:
      effs_errs = np.take(effs_errs, idx)
    else:
      effs_errs = np.zeros(opt_effs.shape)
	
    
    if rates_errs is not None:
      rates_errs = np.take(rates_errs, idx)
    else:
      rates_errs = np.zeros(uniq_rates.shape)    
    return (np.array(opt_effs), np.array(sorted(list(uniq_rates))),
            effs_errs, rates_errs)

def plot_eff_rate(effs, rates, outdir, model, effs_errs=None, rates_errs=None, 
                  xlabel=None, ylabel=None, legend=False, **kwargs):
  
  effs, rates, effs_errs, rates_errs = optimal_eff_rate(effs, rates, effs_errs, rates_errs)

  fig, ax = plt.subplots(figsize=(8,7))  
  plot = 'plot'
  if effs_errs is not None or rates_errs is not None:
    plot = 'errorbar'
    
  if effs_errs is None:
    effs_errs = np.zeros(effs.shape)
      
  if rates_errs is None:
    rates_errs = np.zeros(rates.shape)
        
        
  if plot == 'plot':
    ax.plot(rates, effs, **kwargs)
  elif plot == 'errorbar':
    ax.errorbar(x=rates, y=effs, xerr=rates_errs, yerr=effs_errs, 
                **kwargs)
  
  ax.set_xscale("log")
  ax.set_ylim((0, 1))     
  ax.set_xlim((1,2*10**3)) 

  if xlabel is None:
    ax.set_xlabel('Trigger rate (kHz)')
  else:
    ax.set_xlabel(xlabel)
              
  if ylabel is None:
    ax.set_ylabel('Signal efficiency')
  else:
    ax.set_ylabel(ylabel)         
  if legend:
    ax.legend()
  plt.title(f"{model}",loc='center') 
  plt.savefig(f"{outdir}/eff_rates.png", dpi=200)
    
def get_bkg_score(scores):   
    bkg_idx = int(len(scores) / 2)
    bkg_scores = scores[bkg_idx:]
    return bkg_scores

def get_sig_score(X_test, y_test, model):   
    sig_ind = y_test.shape[0] / 2 
    sig_test = X_test[0:int(sig_ind)]
    sig_score, _ = predict(model, sig_test) 
    return sig_score

def efficiency(score,threshold):
    tpr = (score[score>threshold].shape[0]) / score.shape[0]
    return tpr

def get_precision(tn,fp,fn,tp):
    precision = tp / (tp + fp)
    return precision

def get_efficiency(tn,fp,fn,tp):
    eff = tp / (tp + fn)
    return eff

def get_f1score(precision, efficiency):
    f1score = 2 / ((1/precision) + (1/efficiency))
    return f1score

def predict(model, X_test, verbose = False):
    from tensorflow import keras
    model = keras.models.load_model(model)
    if verbose: 
        print(model.summary())
    test_scores = model.predict(X_test)
    return test_scores

def plotter():
    #enter the model name you define in the ParticleNetLite++
    model_name = 'model_test'
    model_path = f'model_record/{model_name}'
    y_test = np.load('inputs/test_labels.npy')
    outdir = f'{model_path}'
    scores = np.load(f'{model_path}/prediction_scores.npy')



    f=open(f"{model_path}/model_metrics.pickle", 'rb')
    metrics = pickle.load(f)
    stdev, volatility = calc_volatility(metrics)
    accuracy_loss(metrics, outdir, model_name)

    

    fpr, tpr, thresholds = roc_curve(list(map(int, list(y_test))), scores) #hacky but roc-curve object expects very specific dtype only
    thresholds = np.linspace(0,1,1001)
    thr_rates = []
    bkg_scores = get_bkg_score(scores)    
    for threshold in thresholds:
        thr_rates.append(bkg_rates(bkg_scores, threshold))
    min_thresh = rate_threshold_curve(thr_rates,thresholds, outdir, model_name)

    tn, fp, fn, tp = confusion_matrix_plotter(scores.round(), list(map(int, list(y_test))), outdir, model_name)

    rates, rates_errs, effs, effs_errs = eff_rate(fpr, tpr, tn, fp, fn, tp, (40*10**3) * (2760 / 3564))
    plot_eff_rate(np.array(effs), np.array(rates), outdir, model_name,
              rates_errs=rates_errs, effs_errs=effs_errs, ls='', capsize=4.)

    # Calculate the performance of the model at thr = 0.5
    accuracy = get_accuracy(tn, fp, fn, tp)
    precision = get_precision(tn,fp,fn,tp)
    efficiency = get_efficiency(tn,fp,fn,tp)
    f1score = get_f1score(precision, efficiency)
    logloss = log_loss(tf.squeeze(y_test), scores)
    stats05 = {'stdev': stdev, 'volatility': volatility, 'accuracy': accuracy, 'precision': precision, 'efficiency': efficiency, 'f1score': f1score,
                'logloss': logloss, 'min_thresh': min_thresh}

    # Calculate the performance of the model at thr = min_thr
    print('Minimum threshold: ', min_thresh)
    scoresthr = np.array(scores - (min_thresh - 0.5)).astype(np.float64)
    tn2, fp2, fn2, tp2 = confusion_matrix_plotter(scoresthr.round(), list(map(int, list(y_test))), outdir, model_path, min_tr=min_thresh)
    accuracy2 = get_accuracy(tn2, fp2, fn2, tp2)
    precision2 = get_precision(tn2,fp2,fn2,tp2)
    efficiency2 = get_efficiency(tn2,fp2,fn2,tp2)
    f1score2 = get_f1score(precision2, efficiency2)
    logloss2 = log_loss(tf.squeeze(y_test), scoresthr)
    stats_minthr = {'stdev': stdev, 'volatility': volatility, 'accuracy': accuracy2, 'precision': precision2, 'efficiency': efficiency2, 'f1score': f1score2,
                'logloss': logloss2, 'min_thresh': min_thresh}
    
    stats = [stats05, stats_minthr]
    print('Saving stats ...\n')
    f = open(f'{outdir}/model_stats.pickle', 'wb')
    pickle.dump(stats, f)
    f.close()

    print(f"saving to: {outdir}")
    return stats

if __name__ == "__main__":
   stats = plotter()
   #print('At 0.5: ', stats[0].get('accuracy'), stats[0].get('efficiency'), stats[0].get('logloss'))
   print(f'Accuracy: {stats[1].get("accuracy"):.4f}')
   print(f'Efficiency: {stats[1].get("efficiency"):.4f}')
   print(f'Test loss: {stats[1].get("logloss"):.4f}')
   print(f'Volatility: {stats[1].get("volatility"):.4f}')