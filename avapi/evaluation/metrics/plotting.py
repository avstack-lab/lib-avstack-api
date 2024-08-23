# from .assignment import get_ap_from_results, get_lamr_from_results, get_prec_rec_from_results


# def plot_all_results_statistics(results: list, by_class: bool = True):
#     ap, mprec, mrec = get_ap_from_results(results, by_class=by_class)
#     lamr, mr, fppi = get_lamr_from_results(results, by_class=by_class)
#     n_files = len(results)
#     if by_class:
#         n_gt = {
#             k: sum(
#                 [
#                     res.confusion_by_class[k][0, 0] + res.confusion_by_class[k][0, 1]
#                     for res in results
#                     if k in res.confusion_by_class
#                 ]
#             )
#             for k in ap.keys()
#         }
#     else:
#         n_gt = sum([res.confusion[0, 0] + res.confusion[0, 1] for res in results])
#     barplot_gt(n_gt, n_files)
#     barplot_ap(ap)
#     barplot_lamr(lamr)


# def plot_ground_truth_statistics():
#     pass


# # def plot_tp_fp(tp, fp, by_class: bool):
# #     import matplotlib.pyplot as plt
# #     plt.barh(
# #         range(n_classes),
# #         fp_sorted,
# #         align="center",
# #         color="crimson",
# #         label="False Positive",
# #     )
# #     plt.barh(
# #         range(n_classes),
# #         tp_sorted,
# #         align="center",
# #         color="forestgreen",
# #         label="True Positive",
# #         left=fp_sorted,
# #     )


# def plot_prec_rec(prec, rec, by_class: bool):
#     """Plots the precision-recall curves for each class"""


# def barplot_gt(gt, n_files: int, **kwargs):
#     """Barplots number of ground truths for all classes"""
#     import matplotlib.pyplot as plt

#     title = "Number of Ground Truths - {} files".format(n_files)
#     xlabel = "Number of objects"
#     fig, ax = _base_bar_plot(gt, xlabel, title, is_int=True, **kwargs)
#     plt.show()
#     return fig


# def barplot_lamr(lamr, **kwargs):
#     """Barplots lamr for classes"""
#     import matplotlib.pyplot as plt

#     title = "LAMR of all classes"
#     xlabel = "log average miss rate"
#     fig, ax = _base_bar_plot(lamr, xlabel, title, **kwargs)
#     ax.set_xlim([0.0, 1.0])
#     plt.show()
#     return fig


# def barplot_ap(ap, **kwargs):
#     """Barplots mAP for classes"""
#     import matplotlib.pyplot as plt

#     title = "mAP of all classes"
#     xlabel = "mAP"
#     fig, ax = _base_bar_plot(ap, xlabel, title, min_bar_show=0.01, **kwargs)
#     ax.set_xlim([0.0, 1.0])
#     plt.show()
#     return fig


# def _base_bar_plot(
#     vals,
#     xlabel,
#     title,
#     total_height=0.8,
#     single_height=1,
#     write_number=True,
#     min_bar_show=-np.inf,
#     is_int=False,
#     figsize=(7, 5),
#     fontsize=14,
#     reverse_subbars=False,
#     reverse_bars=False,
#     color_bias=0,
#     color_squeeze=1.0,
#     cmap="RdYlGn",
# ):
#     """Base bar plot function for all metrics

#     vals will ultimately be a list[dict] where the
#     list will represent different runs and the dict
#     keys represent different object classes
#     """

#     import matplotlib.pyplot as plt

#     # input wrapping...for modularity's sake :)
#     full_spec = (
#         isinstance(vals, list)
#         and isinstance(vals[0], tuple)
#         and isinstance(vals[0][0], str)
#         and isinstance(vals[0][1], dict)
#     )
#     if not full_spec:
#         raise NotImplementedError("handle different input cases later...")

#     # helpers for bar width etc
#     keys = sorted({key for _, vals_run in vals for key in vals_run.keys()})
#     kloops = keys if not reverse_bars else reversed(keys)
#     n_runs = len(vals)
#     bar_height = total_height / n_runs

#     # make the plot
#     fig, ax = plt.subplots(figsize=figsize)
#     colors = plt.get_cmap(cmap)(
#         color_bias + color_squeeze * np.linspace(0.0, 0.85, len(vals))
#     )

#     # outer loop is over runs
#     handles = []
#     labels = []
#     i = -1
#     vals_loop = vals if not reverse_subbars else reversed(vals)
#     for color, (vtitle, vals_run) in zip(colors, vals_loop):
#         i += 1
#         first = True
#         y_offset = (i - n_runs / 2) * bar_height + bar_height / 2
#         # inner loop is over classes
#         for i_k, k in enumerate(kloops):
#             if k in vals_run:
#                 bars = ax.barh(
#                     y=i_k + y_offset,
#                     width=max(vals_run[k], min_bar_show),
#                     left=0,
#                     height=bar_height * single_height,
#                     color=color,
#                 )
#                 if first:
#                     handles.append(bars)
#                     labels.append(vtitle)
#                     first = False
#                 if write_number:
#                     ax.bar_label(
#                         bars,
#                         labels=[f"{vals_run[k]:d}" if is_int else f"{vals_run[k]:.2f}"],
#                         fontsize=int(fontsize * 0.9),
#                     )
#     # set additional plot parameters
#     ax.set_yticks(range(len(keys)))
#     ax.set_yticklabels(kloops)
#     ax.legend(
#         reversed(handles),
#         reversed(labels),
#         fancybox=True,
#         shadow=True,
#         loc="upper center",
#         bbox_to_anchor=(0.60, 0.08),
#     )
#     ax.set_xlabel(xlabel)
#     ax.xaxis.set_label_position("top")
#     ax.xaxis.set_ticklabels([])
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
#     # ax.set_title(title)
#     # set plot fonts
#     for item in (
#         [ax.yaxis.label]
#         + ax.get_xticklabels()
#         + ax.get_yticklabels()
#         + ax.get_legend().get_texts()
#     ):
#         item.set_fontsize(fontsize)
#     ax.title.set_fontsize(int(fontsize * 1.2))
#     ax.xaxis.label.set_fontsize(int(fontsize * 1.4))
#     return fig, ax
