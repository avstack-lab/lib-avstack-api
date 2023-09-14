import matplotlib.pyplot as plt
import numpy as np
import math


# =============================================
# Precision-Recall Utilities
# =============================================

def precision(confusion):
    if (confusion[0,0] + confusion[1,0]) > 0:
        return confusion[0,0] / (confusion[0,0] + confusion[1,0])
    else:
        return None


def recall(confusion):
    if (confusion[0,0] + confusion[0,1]) > 0:
        return confusion[0,0] / (confusion[0,0] + confusion[0,1])
    else:
        return None


def get_prec_rec_from_results(results: list, by_class: bool):
    if by_class:
        prec, rec = {}, {}
    else:
        prec, rec = [], []
    for res in results:
        p, r = res.get_prec_rec(by_class=by_class)
        if by_class:
            for k in p:
                if k not in prec:
                    prec[k] = []
                    rec[k] = []
                prec[k].append(p[k])
                rec[k].append(r[k])
    return prec, rec


def get_ap_from_results(results: list, by_class: bool):
    prec, rec = get_prec_rec_from_results(results, by_class=by_class)
    if by_class:
        ap, mprec, mrec = {}, {}, {}
        for k in prec:
            ap[k], mprec[k], mrec[k] = average_precision(prec[k], rec[k])
    else:
        ap, mprec, mrec = average_precision(prec, rec)
    return ap, mprec, mrec


def get_lamr_from_results(results: list, by_class: bool):
    prec, rec = get_prec_rec_from_results(results, by_class=by_class)
    if by_class:
        lamr, mr, fppi = {}, {}, {}
        for k in prec:
            lamr[k], mr[k], fppi[k] = log_average_miss_rate(prec[k], rec[k])
    else:
        lamr, mr, fppi = log_average_miss_rate(prec, rec)
    return lamr, mr, fppi


def average_precision(prec: list, rec: list):
    prec = [p for p in prec if p is not None]
    rec = [r for r in rec if r is not None]
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        try:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        except IndexError:
            pass
    return ap, mpre, mrec


def log_average_miss_rate(prec: list, rec: list):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
            State of the Art." Pattern Analysis and Machine Intelligence, IEEE
            Transactions on 34.4 (2012): 743 - 761.
    """
    prec = [p for p in prec if p is not None]
    rec = [r for r in rec if r is not None]
    prec = np.asarray(prec)
    rec = np.asarray(rec)

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


# =============================================
# Metrics Plotting Utilities
# =============================================

def plot_all_results_statistics(results: list, by_class: bool=True):
    ap, mprec, mrec = get_ap_from_results(results, by_class=by_class)
    lamr, mr, fppi = get_lamr_from_results(results, by_class=by_class)
    n_files = len(results)
    if by_class:
        n_gt = {k:sum([res.confusion_by_class[k][0,0] + res.confusion_by_class[k][0,1] for res in results  if k in res.confusion_by_class]) for k in ap.keys()}
    else:
        n_gt = sum([res.confusion[0,0] + res.confusion[0,1] for res in results])
    barplot_gt(n_gt, n_files)
    barplot_ap(ap)
    barplot_lamr(lamr)


def plot_ground_truth_statistics():
    pass


def plot_tp_fp(tp, fp, by_class: bool):
    plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
    plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)


def plot_prec_rec(prec, rec, by_class: bool):
    """Plots the precision-recall curves for each class"""
    pass


def barplot_gt(gt, n_files: int):
    """Barplots number of ground truths for all classes"""
    title = 'Number of Ground Truths - {} files'.format(n_files)
    xlabel = 'Number of objects'
    fig, ax = _base_bar_plot(gt, xlabel, title, is_int=True)
    plt.show()


def barplot_lamr(lamr):
    """Barplots lamr for classes"""
    title = 'LAMR of all classes'
    xlabel = 'log average miss rate'
    fig, ax = _base_bar_plot(lamr, xlabel, title)
    ax.set_xlim([0.0,1.0])
    plt.show()


def barplot_ap(ap):
    """Barplots mAP for classes"""
    title = 'mAP of all classes'
    xlabel = 'mAP'
    fig, ax = _base_bar_plot(ap, xlabel, title)
    ax.set_xlim([0.0,1.0])
    plt.show()


def _base_bar_plot(vals, xlabel, title, total_height=0.8, single_height=1, write_number=True, is_int=False):
    """Base bar plot function for all metrics
    
    vals will ultimately be a list[dict] where the 
    list will represent different runs and the dict
    keys represent different object classes
    """

    # input wrapping...for modularity's sake :)
    full_spec = isinstance(vals, list) and \
                isinstance(vals[0], tuple) and \
                isinstance(vals[0][0], str) and \
                isinstance(vals[0][1], dict)
    if not full_spec:
        raise NotImplementedError("handle different input cases later...")

    # helpers for bar width etc
    keys = {key for _, vals_run in vals for key in vals_run.keys()}
    n_runs = len(vals)
    bar_height = total_height / n_runs

    # make the plot
    fig, ax = plt.subplots(figsize=(9.2, 5))
    colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(vals)))

    # outer loop is over runs
    handles = []
    labels = []
    i = -1
    for color, (vtitle, vals_run) in zip(colors, vals):
        i += 1
        first = True
        y_offset = (i - n_runs / 2) * bar_height + bar_height / 2
        # inner loop is over classes
        for i_k, k in enumerate(reversed(sorted(keys))):
            if k in vals_run:
                bars = ax.barh(
                    y=i_k+y_offset,
                    width=vals_run[k],
                    left=0,
                    height=bar_height*single_height,
                    color=color
                )
                if first:
                    handles.append(bars)
                    labels.append(vtitle)
                    first = False
                if write_number:
                    ax.bar_label(bars, fmt='%d' if is_int else '%.2f')
    # set additional plot parameters
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(reversed(sorted(keys)))
    ax.legend(handles, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return fig, ax
