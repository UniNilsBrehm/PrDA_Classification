# Clustering Analysis
## Packages
```shell
numpy
IPython
pandas
matplotlib
seaborn
scipy
sklearn
joblib
pickle
```

This is the order you should use the scripts:

1. Metadata
'get_meta_data_from_sweeps.py'
Collect all metadata information from the Ca Imaging recoding data, ventral root recordings and stimulus log files
creates: meta_data.csv and sampling_rate.csv

2. Matching Sampling Rates
'match_sampling_rates.py'
Match the sampling rates of some recordings to the one of all the other recordings

3. Data Selection
'select_good_sweeps.py'
Select good recordings for further analysis and convert it to dF/F

4. Stimulus
'stimulus_pre_processing.py'
Prepare stimulus data for further analysis
 
5. Ventral Root: Preprocessing
'ventral_root_pre_processing.py'
Prepare ventral root recordings (converting Olympus .txt files) for further analysis

6. Ventral Root: Time Alignment
'ventral_root_align_recordings_with_ca_imaging.py'
Time align ventral root recordings with ca imaging data

7. Ventral Root: Event Detection
'ventral_root_event_detection.py'
Detect swim bouts in the ventral root traces

8. Linear Scoring of Responses to Stimuli and Motor Events
'linear_scoring_analysis.py'
Using a linear regression model to score ROI responses.
Score = R² * slope

Optional:
- Remove Motor Events from Ca Responses
  "remove_motor_events_from_ca_responses.py"

   


### ----------
Nils Brehm - 2024
