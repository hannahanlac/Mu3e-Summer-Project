import pandas as pd
import re

test_data = pd.read_parquet('ProcessedData/signal1_96_32652/test_data/test_data.parquet')

# Aggregate test data by Unique TID
grouped_test = test_data.groupby("tid_array").agg(
    tid_p=("traj_p", "first"),         # Assuming same p for all hits in a track
    tid_pt=("traj_pt", "first"),       # Same assumption
    tid_lambda=("traj_lambda", "first"),
    tid_phi=("traj_phi", "first"),
    num_hits=("tid_array", "count")    # Count of hits per track
).reset_index()

grouped_test = grouped_test.rename(columns={'tid_array': 'tid'})

#####

reconstructed_tracks = pd.read_csv('ProcessedData/signal1_96_32652/reconstructed_tracks/predicted_tracks3.csv')

# Drop rows where correct_track is 0
reconstructed_tracks = reconstructed_tracks[reconstructed_tracks['correct_track'] != 0]

# Remove non-numeric characters and collapse 'tids' into a single integer value
reconstructed_tracks['tids'] = reconstructed_tracks['tids'].apply(lambda x: int(re.sub(r'\D', '', x)))

# Rename column
reconstructed_tracks = reconstructed_tracks.rename(columns={'tids': 'tid'})

# Merge the dataframes on 'tid'
merged_data = pd.merge(grouped_test, reconstructed_tracks, on='tid', how='left', indicator='merged')
merged_data['merged'] = merged_data['merged'].apply(lambda x: 1 if x == 'both' else 0)

# Save the merged data to a new file
merged_data.to_csv('ProcessedData/signal1_96_32652/evaluation_prep/merged_evaluation1.csv', index=False)
