import pandas as pd
import ast

###################### Predicted Tracks - Condition 1 ##########################

# Load the CSV file
file_path = 'ProcessedData/signal1_96_32652/reconstructed_tracks/predicted_tracks4.csv'
print("Loading CSV file...")
df = pd.read_csv(file_path)

# Define the function to check the percentage of hits belonging to the same tid
def check_correct_tracks(tids_str, threshold, inclusive=False):
    tids = ast.literal_eval(tids_str)
    tid_counts = pd.Series(tids).value_counts()
    max_count = tid_counts.max()
    if inclusive:
        correct_track = 1 if max_count / len(tids) >= threshold else 0
    else:
        correct_track = 1 if max_count / len(tids) > threshold else 0
    return correct_track, max_count

# Define the threshold percentage
threshold_mapping = {
    '100': (1.00, True),
    '75': (0.75, False),
    '50': (0.50, False)
}
command = '50'  # Change this to '100', '75', or '50' as needed
threshold, inclusive = threshold_mapping[command]

print(f"Applying threshold {command} with inclusivity: {inclusive}")

# Apply the function to each row and create new columns 'correct_track' and 'num_tid_hits'
df[['correct_track', 'num_tid_hits']] = df['tids'].apply(lambda x: check_correct_tracks(x, threshold, inclusive)).apply(pd.Series)

print(f"Number of tracks before filtering: {len(df)}")
# Drop rows where 'correct_track' is 0
df = df[df['correct_track'] == 1]
print(f"Number of tracks after filtering: {len(df)}")

print("Flattening TIDs...")
# Update the 'tids' column to a single integer based on the given conditions
def update_tid_column(tids_str, threshold, inclusive=False):
    tids = ast.literal_eval(tids_str)
    tid_counts = pd.Series(tids).value_counts()
    if inclusive:
        return tid_counts.index[0]  # All tids are the same
    else:
        max_tid = tid_counts.idxmax()
        max_count = tid_counts.max()
        if max_count / len(tids) > threshold:
            return max_tid
        else:
            return None

# Apply the update_tid_column function
df['tid'] = df['tids'].apply(lambda x: update_tid_column(x, threshold, inclusive))

# Drop the original 'tids' column
df = df.drop(columns=['tids'])

####################### Test Data File #############################

# Load the test data parquet file and keep only the specified columns
test_data_file_path = 'ProcessedData/signal1_96_32652/test_data/test_data.parquet'
print("Loading test data parquet file...")
test_data_df = pd.read_parquet(test_data_file_path, columns=["frame_array", "tid_array", "traj_p", "traj_pt", "traj_lambda", "traj_phi"])

# Calculate num_hits_x for each tid_array
test_data_df['num_hits_x'] = test_data_df.groupby('tid_array')['tid_array'].transform('count')

# Keep only unique tids and drop rows with less than 4 hits
test_data_df = test_data_df[test_data_df['num_hits_x'] >= 4].drop_duplicates(subset=['tid_array'])

# Sort the test data by unique tid_array
sorted_test_data_df = test_data_df.sort_values(by="tid_array")

####################### Merging Test File and Predicted Tracks #################################

print("Merging data files on tid...")
# Merge the datasets on the tid_array column
merged_df = pd.merge(sorted_test_data_df, df, left_on="tid_array", right_on="tid", how="left")

merged_df = merged_df.drop(columns=['tid'])

# Rename columns to match the desired output
merged_df.rename(columns={'tid_array': 'tid'}, inplace=True)

# Calculate num_hits_y as the number of hits in hit_IDs
merged_df['num_hits_y'] = merged_df['hit_IDs'].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) else 0)

# Debug print to check columns after merge
print("Columns after merge:", merged_df.columns)

# Ensure the columns exist before attempting to fill missing values
merged_df['hit_IDs'] = merged_df['hit_IDs'].fillna('[]')
merged_df['num_hits_x'] = merged_df['num_hits_x'].fillna(0)
merged_df['num_hits_y'] = merged_df['num_hits_y'].fillna(0)
merged_df['correct_track'] = merged_df['correct_track'].fillna(0)
merged_df['num_tid_hits'] = merged_df['num_tid_hits'].fillna(0)

# Drop unnecessary columns
columns_to_keep = ["frame_array", "tid", "num_hits_x", "traj_p", "traj_pt", "traj_lambda", "traj_phi", "hit_IDs", "num_hits_y", "num_tid_hits", "correct_track"]
merged_df = merged_df[columns_to_keep]

print("Columns after filter:", merged_df.columns)

# Convert columns to appropriate data types
merged_df['num_hits_x'] = merged_df['num_hits_x'].astype(int)
merged_df['num_hits_y'] = merged_df['num_hits_y'].astype(int)
merged_df['correct_track'] = merged_df['correct_track'].astype(int)
merged_df['num_tid_hits'] = merged_df['num_tid_hits'].astype(int)

# Check the size of the DataFrame
print(f"Size of merged DataFrame: {merged_df.shape}")

print("Saving CSV...")
# Save the final DataFrame to a new CSV file
output_file_path = f'ProcessedData/signal1_96_32652/evaluation_prep/merged_tracks_{command}.csv'
merged_df.to_csv(output_file_path, index=False)

print(f"Merged CSV file saved to {output_file_path}")