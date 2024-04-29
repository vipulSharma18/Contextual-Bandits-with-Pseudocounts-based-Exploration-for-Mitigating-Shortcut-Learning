# total images: 
# 102000 = (3200+1000+900)*(5 vals of alpha_s)*(4 vals of p_s)

# class +1 or -1
# output class-balanced dataset
# 3200 train instances, 
# 1000 validation & -> 4200 sampled from gaussian directly
# 900 probe/test instances

# for probe generation, paper does uniform sampling
# uniform sampling in the (z_s, z_c) space 
# cartesian product of 30 z_s evenly sampled in [-3u_s, +3u_s]
# & 30 z_c in [-3u_c, +3u_c]

# d = 100 (dimensionality of vectors)
# n_c, n_s = 0
# p_c = 0.9 and \sigma or std_dev_{sc} = 0.6
# 0.5 < p_s < p_c = 0.9 -> 0.6, 0.7, 0.8 and 0.9

#μi = √2 erf^{−1} (2ρi − 1)

# \alpha_c = 1, \alpha_s in [1,5] integer
# based footprint of z_s is 400 pixels

# color is given by raw feature value, amplification and embedding only change availability 
import pandas as pd
import numpy as np
import os 
import gc

from generative_process import embed, combine_features, weights_for_features, sample_z, generate_test


#embed(feature, alpha, weight, nesting_constant)
#combine_features(embed_s, embed_c)
#weights_for_features(dimension=100)
#sample_z(class_label, p_s=0.6, p_c=0.9, std_sc=0.6, samples=2100)
#generate_test(p_s=0.6, p_c=0.9, samples=900)
def count_rows(filename):
    with open(filename, 'r') as file:
        num_rows = sum(1 for line in file)
    return num_rows

def cache_data(cache_path, data_to_cache):
    data_to_cache.to_csv(cache_path, mode='a', header=not os.path.exists(cache_path), index=False)

#amplification is independent of predictivity (z sampling) & of class, hence weights are independent of predictivity and class. 
#otherwise we won't be able to keep 1 constant while varying the other.
classes = [+1, -1]
dimensions = 100
predictivities_s = [0.6, 0.7, 0.8, 0.9]
predictivities_c = [0.9]
w_s = weights_for_features(dimension=dimensions)
w_c = weights_for_features(dimension=dimensions)

#generate train and val data: 
std_sc = 0.6
samples = 2100
alpha_c = 1
alpha_s = [1,2,3,4,5]

synthetic_data_path = 'synthetic_train_data.csv'

for class_label in classes:     
    for p_c in predictivities_c: 
        for p_s in predictivities_s:             
            z = sample_z(class_label, p_s, p_c, std_sc, samples=2100)
            z_s = z[:, 0, :]
            z_c = z[:, 1, :]
            embed_c = embed(z_c, alpha_c, w_c, nesting_constant=0)
            for a_s in alpha_s: 
                embed_s = embed(z_s, a_s, w_s, nesting_constant=0)
                x = combine_features(embed_s, embed_c)
                df = pd.DataFrame({'class_label': [class_label]*x.shape[0], 
                               'p_c': [p_c]*x.shape[0], 
                               'p_s': [p_s]*x.shape[0],
                               'a_c': [alpha_c]*x.shape[0],
                               'a_s': [a_s]*x.shape[0],
                               'z_s': z_s.tolist(), 
                               'z_c': z_c.tolist(),
                               'w_s': [w_s]*x.shape[0],
                               'w_c': [w_c]*x.shape[0],
                               'embed_c': embed_c.tolist(),
                               'embed_s': embed_s.tolist(),
                               'x': x.tolist()})
                cache_data(synthetic_data_path, df)
                del df
                gc.collect()
print("Generated train rows: ", count_rows(synthetic_data_path))

#test data generation
synthetic_data_path = 'synthetic_test_data.csv'
for p_c in predictivities_c: 
    for p_s in predictivities_s: 
        z = generate_test(p_s=p_s, p_c=p_c, samples=900)
        z_s = z[:, 0, :]
        z_c = z[:, 1, :]
        x = combine_features(z_s, z_c)
        df = pd.DataFrame({'p_c': [p_c]*x.shape[0], 
                            'p_s': [p_s]*x.shape[0],
                            'z_s': z_s.tolist(), 
                            'z_c': z_c.tolist(),
                            'x': x.tolist()})
        cache_data(synthetic_data_path, df)
        del df
        gc.collect()
print("Generated test rows: ", count_rows(synthetic_data_path))