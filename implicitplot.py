from implicitpandas import Recommender
import seaborn as sns
import pandas as pd


def build_recommender(data, more_than=1):
    vc = data.userid.value_counts()
    usercount = pd.DataFrame({'userid': vc.index, 'n': vc.values})
    topusers = usercount[usercount.n > more_than]
    topdata = pd.merge(data, topusers, on="userid")
    r = Recommender(factors=2)
    r.train(topdata)
    return r


def plot_users(r, sample=3000, kind="hex"):
    # kind: { "scatter" | "reg" | "resid" | "kde" | "hex" }
    userf = r.user_factors()
    sns.jointplot("f0", "f1", data=userf.sample(sample), kind=kind)


def plot_items(r, sample=3000, kind="hex"):
    # kind: { "scatter" | "reg" | "resid" | "kde" | "hex" }
    itemf = r.item_factors()
    sns.jointplot("f0", "f1", data=itemf.sample(sample), kind=kind)
