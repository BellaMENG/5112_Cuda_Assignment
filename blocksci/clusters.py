import blocksci
import pandas as pd

chain = blocksci.Blockchain("/mnt/storage1/zmengaa/data/blocksci.config")
cm = blocksci.cluster.ClusterManager("/mnt/storage1/zmengaa/data/blocksci_data/clusters", chain)

print("Number of Clusters:", len(cm.clusters()))

addr_lens = []
for cluster in cm.clusters():
    addr_len = cluster.address_count()
    addr_lens.append(addr_len)

df = pd.DataFrame(addr_lens, columns=['address_length'])
max_value = df['address_length'].max()
min_value = df['address_length'].min()

print("The mean of cluster sizes is", df['address_length'].mean())
print("The median of cluster sizes is", df['address_length'].median())
print("The largest cluster contains", max_value, "addresses.")
print("The smallest cluster contains", min_value, "addresses.")
