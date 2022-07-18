# cross-sessions_RS
This is the data and source code for our paper **Learning Recommendations from User Actions in the Item-poor Insurance Domain**.

## Requirements

- Python
- NumPy
- Pandas
- TensorFlow
- Scikit-learn
- Pickle
- Statsmodels
- Bioinfokit
- SciPy
- Matplotlib


## Dataset

We publish an anonymized version of the real-world dataset from the insurance domain used to implement all the session-based recommender models.
Download the 6 files: sessions_train.csv, sessions_test.csv, purchase_events_train.csv, purchase_events_test.csv, filter_train.csv, filter_test.csv

Due to privacy protection, we cannot publish the portfolios and demographic attributes used to implement the insurance baseline models SVD and demographic.


## Usage

1. Train and validate the models using  
   SVD.py  
   demographic.py  
   GRU4REC.py  
   GRU4REC_concat.py  
   SKNN_E.py  
   SKNN_EB.py  
   cross_sessions_encode.py  
   cross_sessions_concat.py  
   cross_sessions_auto.py  
2. Evaluate the models over the test set using  
   random_evaluation.py  
   popular_evaluation.py  
   SVD_evaluation.py  
   demographic_evaluation.py  
   GRU4REC_evaluation.py  
   GRU4REC_concat_evaluation.py  
   SKNN_E_evaluation.py  
   SKNN_EB_evaluation.py  
   cross_sessions_encode_evaluation.py  
   cross_sessions_concat_evaluation.py  
   cross_sessions_auto_evaluation.py  
3. Plot evaluation measures for varying thresholds and test for statistical significans using  
   varying_thresholds_plot.py  
   statistical_significans_test.py  
