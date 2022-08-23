import pandas as pd
import numpy as np
from pathlib import Path
import asrtoolkit
from fuzzywuzzy import fuzz
from nltk.metrics import distance
import string
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_results(filenames: list) -> (list, pd.DataFrame):
    results = pd.DataFrame()
    models = list()

    for idx, filename in enumerate(filenames):
        new_results = pd.read_csv(filename)
        model_label = Path(filename).stem  # .split('-')[0]
        ground_truth_label = f'{model_label}-ground_truth'
        new_results = new_results.rename(columns={'label': ground_truth_label,
                                                  'prediction': model_label})
        del new_results['Unnamed: 0']
        models.append((ground_truth_label, model_label))
        new_results[ground_truth_label] = new_results.apply(lambda r: str(r[ground_truth_label]), axis=1)
        new_results[model_label] = new_results.apply(lambda r: str(r[model_label]), axis=1)
        results = pd.concat([results, new_results], axis=1)

    models = sorted(models, key=lambda b: b[1])
    return models, results


def generate_summary_stats(models: list, results: pd.DataFrame) -> pd.DataFrame:
    NUM_STATS = 6
    NUM_SUB_STATS = 5
    a = ['matchratio'] * NUM_SUB_STATS + ['CER'] * NUM_SUB_STATS + ['matchratio_insensitive'] * NUM_SUB_STATS + \
        ['CER_insensitive'] * NUM_SUB_STATS + ['matchratio_nopunc'] * NUM_SUB_STATS + ['CER_nopunc'] * NUM_SUB_STATS
    b = ['mean', 'min', 'max', 'stdev', 'median'] * NUM_STATS
    headers = pd.MultiIndex.from_arrays([a, b], names=('stat', 'substats'))
    summary_results = pd.DataFrame(columns=headers)

    def _apply_stat(model_label: str, query_label: str, summary_stat_label: str, stat_function):
        results[query_label] = subset.apply(stat_function, axis=1)
        summary_results.loc[model_label, (summary_stat_label, 'mean')] = results[query_label].mean()
        summary_results.loc[model_label, (summary_stat_label, 'stdev')] = results[query_label].std()
        summary_results.loc[model_label, (summary_stat_label, 'min')] = results[query_label].min()
        summary_results.loc[model_label, (summary_stat_label, 'median')] = results[query_label].median()
        summary_results.loc[model_label, (summary_stat_label, 'max')] = results[query_label].max()
        subset[query_label] = results[query_label]

    for ground_truth_label, model_label in models:
        subset = results[[ground_truth_label, model_label]].dropna()

        _apply_stat(model_label, f'{model_label} CER', 'CER',
                    lambda r: asrtoolkit.cer(str(r[ground_truth_label]), str(r[model_label])))
        _apply_stat(model_label, f'{model_label} match ratio', 'matchratio',
                    lambda r: fuzz.ratio(str(r[ground_truth_label]), str(r[model_label])))
        results[f'{model_label}-edit_distance'] = subset.apply(
            lambda r: distance.edit_distance(str(r[ground_truth_label]), str(r[model_label])), axis=1)

        subset[f'{ground_truth_label}-lower'] = subset[ground_truth_label].map(str.lower)
        subset[f'{model_label}-lower'] = subset[model_label].map(str.lower)
        _apply_stat(model_label, f'{model_label} CER insensitive', 'CER_insensitive',
                    lambda r: asrtoolkit.cer(str(r[f'{ground_truth_label}-lower']), str(r[f'{model_label}-lower'])))
        _apply_stat(model_label, f'{model_label} match ratio insensitive', 'matchratio_insensitive',
                    lambda r: fuzz.ratio(str(r[f'{ground_truth_label}-lower']), str(r[f'{model_label}-lower'])))

        subset[f'{ground_truth_label}-nopunc'] = subset[ground_truth_label].map(
            lambda s: s.translate(str.maketrans('', '', string.punctuation)))
        subset = subset.loc[subset[f'{ground_truth_label}-nopunc'].str.len() > 0]
        subset[f'{model_label}-nopunc'] = subset[model_label].map(
            lambda s: s.translate(str.maketrans('', '', string.punctuation)))
        subset = subset.loc[subset[f'{model_label}-nopunc'].str.len() > 0]
        _apply_stat(model_label, f'{model_label} CER no punc', 'CER_nopunc',
                    lambda r: asrtoolkit.cer(str(r[f'{ground_truth_label}-nopunc']), str(r[f'{model_label}-nopunc'])))
        _apply_stat(model_label, f'{model_label} match ratio no punc', 'matchratio_nopunc',
                    lambda r: fuzz.ratio(str(r[f'{ground_truth_label}-lower']), str(r[f'{model_label}-lower'])))

        exact = results.dropna().apply(lambda r: 1 if int(r[f'{model_label}-edit_distance']) == 0 else 0, axis=1).sum()
        oboe = results.dropna().apply(lambda r: 1 if int(r[f'{model_label}-edit_distance']) == 1 else 0, axis=1).sum()
        summary_results.loc[model_label, 'exact_matches'] = pd.to_numeric(exact, downcast='integer')
        summary_results.loc[model_label, 'oboe_matches'] = pd.to_numeric(oboe, downcast='integer')

        err_range = stats.norm.interval(alpha=0.95,
                                        loc=np.mean(results[f'{model_label} CER'].dropna()),
                                        scale=stats.sem(results[f'{model_label} CER'].dropna()))
        summary_results.loc[model_label, 'cer_95_error_min'] = err_range[0]
        summary_results.loc[model_label, 'cer_95_error_max'] = err_range[1]

    return summary_results


def plot_cer(models: list, results: pd.DataFrame, graph) -> None:
    graph.set_title('Character Error Rate (CER)')
    graph.set_xticklabels([label[1] for label in models], rotation=15, horizontalalignment='right')
    graph.set_ylim([-1, 200])
    subset = results[[f'{label[1]} CER' for label in models]].T
    subset = np.asarray([model[1].dropna() for model in subset.iterrows()], dtype=object).T
    graph.boxplot(subset)


def plot_cer_confidence_interval(models: list, summary_results: pd.DataFrame, graph) -> None:
    min = summary_results['cer_95_error_min']
    min = min.apply(lambda v: v if v >= 0 else 0)
    max = summary_results['cer_95_error_max']
    max = max.apply(lambda v: v if v <= 100 else 100)
    error_min = [summary_results.loc[label[1], ('CER', 'mean')] - min[label[1]] for label in models]
    error_max = [max[label[1]] - summary_results.loc[label[1], ('CER', 'mean')] for label in models]
    graph.set_title('CER 95% confidence interval')
    graph.set_xticklabels([label[1] for label in models], rotation=15, horizontalalignment='right')
    graph.set_ylim(top=100)
    graph.errorbar([label[1] for label in models], summary_results[('CER', 'mean')], yerr=[error_min, error_max], fmt='o')
    for i in range(len(models)):
        graph.text(i + 0.05, max[i] - 2, f'{max[i]:.3f}')
        graph.text(i + 0.05, min[i] + 1, f'{min[i]:.3f}')


def plot_fuzzy_match(models: list, results: pd.DataFrame, graph) -> None:
    graph.set_title('Fuzzy Match Ratio')
    graph.set_xticklabels([label[1] for label in models], rotation=15, horizontalalignment='right')
    subset = results[[f'{label[1]} match ratio' for label in models]].T
    subset = np.asarray([model[1].dropna() for model in subset.iterrows()], dtype=object).T
    graph.boxplot(subset)


def plot_close_matches(models: list, results: pd.DataFrame, summary_results: pd.DataFrame, graph):
    graph.set_title('% of Exact/Almost Exact String Matches')
    exact_matches = [match/results.shape[0] for match in summary_results['exact_matches']]
    obo_matches = [match/results.shape[0] for match in summary_results['oboe_matches']]
    ex = graph.bar([label[1] for label in models], exact_matches,
                   width=0.3, label='Exact match')
    obo = graph.bar([label[1] for label in models], obo_matches,
                    width=0.3, bottom=exact_matches, label='Off by one')
    graph.set_xticklabels([label[1] for label in models], rotation=15, horizontalalignment='right')
    graph.set_ybound(0, 0.2)
    graph.legend()
