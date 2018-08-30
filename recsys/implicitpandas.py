from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.utils import nonzeros
import numpy as np
import pandas as pd
import pickle
import os


class RecSysWrapper:
    def load(self, dir):
        self.usrvsart = load(os.path.join(dir, "usrvsart.pkl"))
        self.cprvsart = load(os.path.join(dir, "cprvsart.pkl"))

    def user_recommendation(self, userid, N=10):
        return self.usrvsart.recommendations(userid, N)

    def cart_recommendation(self, itemids, N=10):
        return self.cprvsart.items_recommendations(itemids, N)


class Recommender:
    def __init__(self, factors=50):
        self.model = AlternatingLeastSquares(factors=factors,
                                             regularization=0.01,
                                             dtype=np.float64,
                                             iterations=50)

    def train(self, data):
        userids = data.userid.astype("category")
        itemids = data.itemid.astype("category")

        matrix = coo_matrix((data.confidence.astype('float64'),
                             (itemids.cat.codes.copy(),
                              userids.cat.codes.copy())))
        self.model.fit(matrix)
        self.t_matrix = matrix.T.tocsr()
        self.userid_to_code = dict([(category, code)
                                    for code, category in enumerate(userids.cat.categories)])
        self.itemid_to_code = dict([(category, code)
                                    for code, category in enumerate(itemids.cat.categories)])
        self.usercode_to_id = dict([(code, category)
                                    for code, category in enumerate(userids.cat.categories)])
        self.itemcode_to_id = dict([(code, category)
                                    for code, category in enumerate(itemids.cat.categories)])

    def similar_items(self, itemid, N=10):
        item_code = self.itemid_to_code[itemid]
        similar_codes = self.model.similar_items(item_code, N)
        similar_ids = [(self.itemcode_to_id[code], s)
                       for code, s in similar_codes]
        return pd.DataFrame(similar_ids, columns=["itemid", "similarity"])

    def recommendations(self, userid, N=10):
        user_code = self.userid_to_code[userid]
        user_item_codes = self.model.recommend(user_code, self.t_matrix, N)
        user_item_ids = [(self.itemcode_to_id[code], c)
                         for code, c in user_item_codes]
        return pd.DataFrame(user_item_ids, columns=["itemid", "confidence"])

    def explain(self, userid, itemid):
        user_code = self.userid_to_code[userid]
        item_code = self.itemid_to_code[itemid]
        return self.model.explain(user_code, self.t_matrix, item_code)

    def confidence(self, userid, itemid):
        item_code = self.itemid_to_code[itemid]
        user_code = self.userid_to_code[userid]
        item_factor = self.model.item_factors[item_code]
        user_factor = self.model.user_factors[user_code]
        return item_factor.dot(user_factor)

    def user_factors(self):
        factors = pd.DataFrame(self.model.user_factors).add_prefix("f")
        ids = factors.index.map(lambda code: self.usercode_to_id[code])
        factors.insert(0, "userid", ids)
        return factors

    def item_factors(self):
        factors = pd.DataFrame(self.model.item_factors).add_prefix("f")
        ids = factors.index.map(lambda code: self.itemcode_to_id[code])
        factors.insert(0, "itemid", ids)
        return factors

    def items_recommendations(self, itemids, N=10):
        user_code = 0
        item_codes = [self.itemid_to_code[id] for id in itemids]

        data = [1 for _ in item_codes]
        rows = [0 for _ in item_codes]
        shape = (1, self.model.item_factors.shape[0])
        user_items = coo_matrix(
            (data, (rows, item_codes)), shape=shape).tocsr()

        user_item_codes = self.model.recommend(
            user_code, user_items, N, recalculate_user=True)
        user_item_ids = [(self.itemcode_to_id[code], c)
                         for code, c in user_item_codes]
        return pd.DataFrame(user_item_ids, columns=["itemid", "confidence"])


class InfoRecommender(Recommender):
    def set_user_info(self, usersinfo):
        self.usersinfo = usersinfo

    def set_item_info(self, itemsinfo):
        self.itemsinfo = itemsinfo

    def similar_items(self, itemid, N=10):
        items = super().similar_items(itemid, N)
        return pd.merge(items, self.itemsinfo, on="itemid")

    def recommendations(self, userid, N=10):
        items = super().recommendations(userid, N)
        return pd.merge(items, self.itemsinfo, on="itemid")

    def user_factors(self):
        factors = super().user_factors()
        return pd.merge(factors, self.usersinfo, on="userid")

    def item_factors(self):
        factors = super().item_factors()
        return pd.merge(factors, self.itemsinfo, on="itemid")

    def items_recommendations(self, itemids, N=10):
        items = super().items_recommendations(itemids, N)
        return pd.merge(items, self.itemsinfo, on="itemid")


def create(datafilename):
    data = pd.read_csv(datafilename)
    rec = Recommender()
    rec.train(data)
    return rec


def save(rec, filename):
    output = open(filename, 'wb')
    pickle.dump(rec, output)
    output.close()


def load(filename):
    pfile = open(filename, 'rb')
    rec = pickle.load(pfile)
    pfile.close()
    return rec
