import numpy as np
import pandas as pd
from scipy.special import erfinv

def nesting(input, nesting_constant=0): 
    #no nesting done in these experiments but this is just a 
    # placeholder for future experiments
    # fully connected random net with nesting_constant tanh layers in cascade
    nested_data = input
    return nested_data

def amplification(alpha, weights, input): 
    #e_i = \alpha_i * w_i * z_i
    # w_i is R^d and final output x is also R^d
    amplified_feature = alpha*weights*input[:,np.newaxis]
    return amplified_feature

def embed(feature, alpha, weight, nesting_constant): 
    amplified_feature = amplification(alpha, weight, feature)
    nested_feature = nesting(amplified_feature)
    return nested_feature
    
def combine_features(embed_s, embed_c): 
    return embed_s + embed_c

def weights_for_features(dimension=100): 
    w = np.random.normal(size=dimension)
    w = w/np.linalg.norm(w)
    return w
    
def sample_z(class_label, p_s=0.6, p_c=0.9, std_sc=0.6, samples=2100): 
    cov = [[1, std_sc], [std_sc, 1]]
    u_s = (2**0.5)*erfinv(2*p_s - 1)
    u_c = (2**0.5)*erfinv(2*p_c - 1)
    mean = [class_label*u_s, class_label*u_c]
    z = np.random.multivariate_normal(mean, cov, samples) #samples, 2
    return z

def sample_z_i(p_i, samples=30): 
    u_i = (2**0.5)*erfinv(2*p_i - 1)
    z_i = np.linspace(-3 * u_i, 3 * u_i, num=samples, endpoint=True) #(samples,)
    return z_i 

def generate_test(p_s=0.6, p_c=0.9, samples=900): 
    z_samples = int(samples**0.5)
    z_s = sample_z_i(p_s, z_samples)
    z_c = sample_z_i(p_c, z_samples)
    z = np.array([[zs_i, zc_i] for zs_i in z_s for zc_i in z_c]) #(samples,2)
    return z