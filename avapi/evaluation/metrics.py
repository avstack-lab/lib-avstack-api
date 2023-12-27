import math

import numpy as np


# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

# =============================================
# Precision-Recall Utilities
# =============================================


def precision(confusion):
    if (confusion[0, 0] + confusion[1, 0]) > 0:
        return confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
    else:
        return None


def recall(confusion):
    if (confusion[0, 0] + confusion[0, 1]) > 0:
        return confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
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
        else:
            prec.append(p)
            rec.append(r)
    return prec, rec


def get_ap_from_results(results: list, by_class: bool):
    """Input is a list of the results"""
    prec, rec = get_prec_rec_from_results(results, by_class=by_class)
    if by_class:
        ap, mprec, mrec = {}, {}, {}
        for k in prec:
            ap[k], mprec[k], mrec[k] = average_precision(prec[k], rec[k])
    else:
        ap, mprec, mrec = average_precision(prec, rec)
    return ap, mprec, mrec


def get_lamr_from_results(results: list, by_class: bool):
    """Input is a list of the results"""
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
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        try:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
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

    fppi = 1 - prec
    mr = 1 - rec

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
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


def plot_all_results_statistics(results: list, by_class: bool = True):
    ap, mprec, mrec = get_ap_from_results(results, by_class=by_class)
    lamr, mr, fppi = get_lamr_from_results(results, by_class=by_class)
    n_files = len(results)
    if by_class:
        n_gt = {
            k: sum(
                [
                    res.confusion_by_class[k][0, 0] + res.confusion_by_class[k][0, 1]
                    for res in results
                    if k in res.confusion_by_class
                ]
            )
            for k in ap.keys()
        }
    else:
        n_gt = sum([res.confusion[0, 0] + res.confusion[0, 1] for res in results])
    barplot_gt(n_gt, n_files)
    barplot_ap(ap)
    barplot_lamr(lamr)


def plot_ground_truth_statistics():
    pass


# def plot_tp_fp(tp, fp, by_class: bool):
#     import matplotlib.pyplot as plt
#     plt.barh(
#         range(n_classes),
#         fp_sorted,
#         align="center",
#         color="crimson",
#         label="False Positive",
#     )
#     plt.barh(
#         range(n_classes),
#         tp_sorted,
#         align="center",
#         color="forestgreen",
#         label="True Positive",
#         left=fp_sorted,
#     )


def plot_prec_rec(prec, rec, by_class: bool):
    """Plots the precision-recall curves for each class"""


def barplot_gt(gt, n_files: int, **kwargs):
    """Barplots number of ground truths for all classes"""
    import matplotlib.pyplot as plt

    title = "Number of Ground Truths - {} files".format(n_files)
    xlabel = "Number of objects"
    fig, ax = _base_bar_plot(gt, xlabel, title, is_int=True, **kwargs)
    plt.show()
    return fig


def barplot_lamr(lamr, **kwargs):
    """Barplots lamr for classes"""
    import matplotlib.pyplot as plt

    title = "LAMR of all classes"
    xlabel = "log average miss rate"
    fig, ax = _base_bar_plot(lamr, xlabel, title, **kwargs)
    ax.set_xlim([0.0, 1.0])
    plt.show()
    return fig


def barplot_ap(ap, **kwargs):
    """Barplots mAP for classes"""
    import matplotlib.pyplot as plt

    title = "mAP of all classes"
    xlabel = "mAP"
    fig, ax = _base_bar_plot(ap, xlabel, title, min_bar_show=0.01, **kwargs)
    ax.set_xlim([0.0, 1.0])
    plt.show()
    return fig


def _base_bar_plot(
    vals,
    xlabel,
    title,
    total_height=0.8,
    single_height=1,
    write_number=True,
    min_bar_show=-np.inf,
    is_int=False,
    figsize=(7, 5),
    fontsize=14,
    reverse_subbars=False,
    reverse_bars=False,
    color_bias=0,
    color_squeeze=1.0,
    cmap="RdYlGn",
):
    """Base bar plot function for all metrics

    vals will ultimately be a list[dict] where the
    list will represent different runs and the dict
    keys represent different object classes
    """

    import matplotlib.pyplot as plt

    # input wrapping...for modularity's sake :)
    full_spec = (
        isinstance(vals, list)
        and isinstance(vals[0], tuple)
        and isinstance(vals[0][0], str)
        and isinstance(vals[0][1], dict)
    )
    if not full_spec:
        raise NotImplementedError("handle different input cases later...")

    # helpers for bar width etc
    keys = sorted({key for _, vals_run in vals for key in vals_run.keys()})
    kloops = keys if not reverse_bars else reversed(keys)
    n_runs = len(vals)
    bar_height = total_height / n_runs

    # make the plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap(cmap)(
        color_bias + color_squeeze * np.linspace(0.0, 0.85, len(vals))
    )

    # outer loop is over runs
    handles = []
    labels = []
    i = -1
    vals_loop = vals if not reverse_subbars else reversed(vals)
    for color, (vtitle, vals_run) in zip(colors, vals_loop):
        i += 1
        first = True
        y_offset = (i - n_runs / 2) * bar_height + bar_height / 2
        # inner loop is over classes
        for i_k, k in enumerate(kloops):
            if k in vals_run:
                bars = ax.barh(
                    y=i_k + y_offset,
                    width=max(vals_run[k], min_bar_show),
                    left=0,
                    height=bar_height * single_height,
                    color=color,
                )
                if first:
                    handles.append(bars)
                    labels.append(vtitle)
                    first = False
                if write_number:
                    ax.bar_label(
                        bars,
                        labels=[f"{vals_run[k]:d}" if is_int else f"{vals_run[k]:.2f}"],
                        fontsize=int(fontsize * 0.9),
                    )
    # set additional plot parameters
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(kloops)
    ax.legend(
        reversed(handles),
        reversed(labels),
        fancybox=True,
        shadow=True,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.08),
    )
    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticklabels([])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    # ax.set_title(title)
    # set plot fonts
    for item in (
        [ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
        + ax.get_legend().get_texts()
    ):
        item.set_fontsize(fontsize)
    ax.title.set_fontsize(int(fontsize * 1.2))
    ax.xaxis.label.set_fontsize(int(fontsize * 1.4))
    return fig, ax
