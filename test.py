from numpy.lib.npyio import load
from pandas.io.json import json_normalize
import json
import ConfigSpace as CS
import motpe
import pandas as pd
import matplotlib.pyplot as plt
seed = 1
f = motpe.CnnFromCfg("nas-motpe", seed)
cs = f.make_cs(CS.ConfigurationSpace(seed=seed))


data = pd.read_json('./models/records_history.json')
f_columns = json_normalize(data['f'].to_list())
data.drop(['f'], axis=1)
data = pd.concat([data, f_columns], axis=1)

filtered = data[data['f1'] < -70]
new_f1 = []

for cfg in filtered['x']:
    r = f(cfg, budget=50, save=False, load=False, trial='0')
    new_f1.append(r['f1'])


se = pd.Series(new_f1)
filtered['new_f1'] = se.values

fig = plt.figure(figsize=(8, 6))
# f1s = [fs['f1'] for fs in history['f']]
# f2s = [fs['f2'] for fs in history['f']]
f1s = filtered['new_f1'].to_numpy()
f2s = filtered['f2'].to_numpy()
plt.scatter(f1s, f2s)
plt.title("NAS")
plt.xlabel('f1')
plt.ylabel('f2')
plt.grid()
plt.show()
filtered.to_json("./models/records_history_top_tuned.json")
