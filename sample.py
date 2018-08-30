from recsys.implicitpandas import RecSysWrapper

rec = RecSysWrapper()
rec.load("./model")

userid = "FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg="
print(rec.user_recommendation(userid))

itemids = ["56GgUdZtxpOVaQWIIPK2gFBiyJxYjbC8gPMPU1stoL8=",
           "Rerp71vL127G1162T5Bwgk3D6r/ZtiXDfWvYz+3ljhE="]
print(rec.cart_recommendation(itemids))
