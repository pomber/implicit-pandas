

def count_quantiles(series, q):
    """
    Count how many times each item is repeated 
    and find equal-sized buckets for the counts.
    Parameters
    ----------
    series : ndarray or Series
    q : integer or array of quantiles
    Returns
    --------
    >>> x = pd.Series(['a']*1 + ['b']*2 + ['c']*3 + ['d']*4 + ['e']*5)
    >>> count_quantiles(x, 3)
    (0.999, 2.333]    2
    (2.333, 4.333]    1
    (4.333, 5.0]      2
    dtype: int64
    """
    return pd.qcut(series.value_counts(), q,
                   duplicates="drop").value_counts(sort=False)


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())
