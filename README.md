# Exploration-For-Shortcut-Learning
Features used by ANNs often differ from humans, they prefer more available features (pixel footprint) even if they are less or similarly predictive as the core ones. The project uses exploration &amp; bandits to guide a supervised learning algo to attend to the core features instead of the spurious ones.


## Notes: 
1. Bias is averaged across 5 runs of the models (so total 10 runs since model and optimal classifier)

I'm not using bounding boxes or IOU as the truth values unlike past RL Computer Vision work. 
For every application, there's some ground truth for localization/segmentation/tracking present. 
I don't have any ground truth, "hope" all via exploration. 

Human feedback in other papers which tune vision models using RL.

Foveation counter ->
It's essentially a RL guided cropping in the image space. A much more complicated dropout. 
But does this lead to any "meaningful" features in the deep neural network which correspond to a class's core concepts? 
Not explicitly set up for this. 

Bootstrapping problem: Seed? Where to start foveation process from?
GradCAM as a solution but GradCAM unreliable. 

Possible reward hacking/shortcut:
* just do a crop such that all of image is included. When to stop expanding crop window??
* just always crop the background and never the foreground if background is higly correlated. Back to square one/problem.
Class labels are not enough, need some "meaning"/feature attributes to be compared.
Uses some threshold of probability matching with true label as "good enough" of a match.


Harmonization/Alignment emerging without any exlicit reward/feedback for it.

	
1. generate dataset for all iamges -> Done. Two types of data generated, stick with oen of them. 

2. contrastive learning model to generate feature states

3. supervised learning model baseline with an alpha ration of 4 and predictivity as 0.9 (baseline). -> done, all settings with 5 seeds data available now. 

4. coding for measures of reliance and bias

5. bandit algos made

6. bandit algos use contrastive embedding to decide mask on or off. 
7. supervised learning has an input context to determine which mask to be on or off. 

- Exploration done via pseudo-counts (input state and output 0 or 10

8. bandits tell supervised which mask to be on or off. 