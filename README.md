# cross-sessions_RS
This is the data and source code for our paper [Learning Recommendations from User Actions in the Item-poor Insurance Domain](https://doi.org/10.1145/3523227.3546775).

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
Download the 6 files: purchase_events_train.csv, purchase_events_test.csv, sessions_train.csv, sessions_test.csv, filter_train.csv, filter_test.csv.

Due to privacy protection, we cannot publish the portfolios and demographic attributes used to implement the insurance baseline models SVD and demographic.


## Dataset Format
There are 3 different datasets, each of which is split into training and test sets.

### puchase_events_train.csv and purchase_events_test.csv

This data contains the purchase events. Each event consists of one or more item purchases made by the same user. The data contains 2 columns:
- event_id. The ID of a purchase event.   
- item_id. The ID of an item.   

### sessions_train.csv and sessions_test.csv

This data contains the user sessions that the user made prior to the user's purchase event. Each session consists of multiple actions. The data contains 6 columns:
- event_id. The ID of a purchase event.   
- session_id. The ID of a session.   
- action_time. The time of an action in a session with format "YYYY/MM/DD HH:MM:SS".   
- action_section. The ID of the section of an action.   
- action_object. The ID of the object of an action.   
- action_type. The ID of the type of an action.   

### filter_train.csv and filter_test.csv

This data contains the items that were possible for the user to buy at the time of the user's purchase event. The data contains 2 columns:
- event_id. The ID of a purchase event.   
- item_id. The ID of an item.   


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
