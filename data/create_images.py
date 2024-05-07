# z_s circle
# z_c square 
# use alpha_c and alpha_s to control size/pixel footprint for train & val. 
# test is kept as the baseline footprint/base z values. 
# The baseline footprint is 400px
# color = respective feature values 
# manipulate size of square and circle. 

# random position of circle and square, no overlap
# output image size is 224x224
# base footprint of z_s is 400 pixels

# total images: 
# 102000 = (3200+1000+900)*(5 vals of alpha_s)*(4 vals of p_s)
# train size = (3200+1000)*5*4 = 84,000 -> 42,000 per class
# test size = 900*5*4 = 18,000

#loop over the training and validation set
# 3200 1000 total for single p_s and alpha_s
# 1600 500 for 1 class 

# test
# 900 for single p_s

import pandas as pd
from sample import generate_image

classes = [1, 0]
predictivities_s = [0.6, 0.7, 0.8, 0.9]
alpha_s = [1,2,3,4,5]

def create_image(data_path = None, mode='train'): 
    print(f"Creating images for file: {data_path}")
    df = pd.read_csv(data_path)
    df['class_label'] = df['class_label'].astype(int)
    for p_s in predictivities_s: 
        for a_s in alpha_s: 
            dataset_folder = str(p_s) + '_' + str(a_s)+'/'
            print(f"Creating images for folder {dataset_folder}")
            for class_ in classes: 
                curr_df = df[(df['p_s']==p_s) & (df['a_s']==a_s) & (df['class_label']==class_)]
                if mode=='train': 
                    splits = ['train/', 'val/'] #split logic
                    indices_1600 = curr_df.sample(n=1600, replace=False).index.tolist()
                    indices_remaining = curr_df[~curr_df.index.isin(indices_1600)].sample(n=500, replace=False).index.tolist()
                    indices = [indices_1600, indices_remaining]
                else: 
                    splits = ['test/']
                    indices = [curr_df.index.tolist()]
                for i,split in enumerate(splits): 
                    save_path = dataset_folder+split+str(class_)+'/'
                    write_df = curr_df.loc[indices[i]]
                    write_df['z_s'] = (write_df['z_s'] - write_df['z_s'].mean())/(write_df['z_s'].max() - write_df['z_s'].min()) * (255)
                    write_df['z_c'] = (write_df['z_c'] - write_df['z_c'].mean())/(write_df['z_c'].max() - write_df['z_c'].min()) * (255)
                    print(f'Processing class {class_}, split {split}, num of images {len(write_df)}')
                    total = len(write_df)
                    for index, row in write_df.iterrows(): 
                        # the image creating function call. use apply or something
                        image = generate_image(circle=int(row['z_s']), square=int(row['z_c']), circle_size=row['a_s'])
                        image.save(save_path+str(index)+'.png')
                        if total%100==0:
                            print('remaining', total)
                        total-=1

create_image('synthetic_test_data.csv', 'test')
# create_image('synthetic_train_data.csv', 'train')

