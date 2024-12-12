Need to activate anaconda environment in VS to run code 

Data_Processes : contains all data processing for lyric data from local machine (file is too large to store remotely) 
Runs basic preprocessing such as removing stop-words, punctuaton, and numbers. Encodes target 'genre' and performs tf-idf vectorization on 'lyric' data. 
Splts data into 4 'splits' stratified by genre to tran each composing model. 1-3 are used for 1 model of each type. 4 is used to train the ensemble. 
Wll run at the beginng of each model file to update the data and return it to the program internally. 

decison_trees: trains 3 decision trees with varying parameters. 

log_regs: trains 3 logistic regression models with varied parameters. Saves them as a pkl file in 'pickel_jar'. 

naive_bayes: trains 3 naive bayes classifier models w/ varied params.Saves them as a pkl file in 'pickel_jar'. 

decision_trees: trains 3 decision tree models with with varied parameters. Saves them as a pkl file in 'pickel_jar'. 

stacked_ensemble: creates an ensemble model with a random forrest meta learner. 

test: tests the ensemble w/ testing data sets and reports accuracy. 