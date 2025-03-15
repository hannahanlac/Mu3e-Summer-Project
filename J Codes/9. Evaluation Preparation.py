import pandas as pd
import ast


###################### Predicted Tracks ##########################

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
        return 1 if max_count / len(tids) >= threshold else 0
    else:
        return 1 if max_count / len(tids) > threshold else 0

# Define the threshold percentage
threshold_mapping = {
    '100': (1.00, True),
    '75': (0.75, False),
    '50': (0.50, False)
}
command = '100'  # Change this to '100', '75', or '50' as needed
threshold, inclusive = threshold_mapping[command]

print(f"Applying threshold {command} with inclusivity: {inclusive}")

# Apply the function to each row and create a new column 'correct_tracks'
df['correct_tracks'] = df['tids'].apply(lambda x: check_correct_tracks(x, threshold, inclusive))

print(f"Number of tracks before filtering: {len(df)}")
# Drop rows where 'correct_tracks' is 0
df = df[df['correct_tracks'] == 1]
print(f"Number of tracks after filtering: {len(df)}")


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

# Sort the test data by unique tid_array
sorted_test_data_df = test_data_df.sort_values(by="tid_array")

print("Merging data files on tid...")
# Merge the datasets on the tid_array column
merged_df = pd.merge(sorted_test_data_df, df, left_on="tid_array", right_on="tid", how="left")

# Rename columns to match the desired output
merged_df.rename(columns={'tid_array': 'tid'}, inplace=True)

# Ensure the columns exist before attempting to fill missing values
if 'hit_IDs' not in merged_df:
    merged_df['hit_IDs'] = '[]'
if 'num_hits_x' not in merged_df:
    merged_df['num_hits_x'] = 0
if 'num_hits_y' not in merged_df:
    merged_df['num_hits_y'] = 0

# Fill NaN values in the merged DataFrame with default values
# Fill NaN values in the merged DataFrame with default values
merged_df['hit_IDs'] = merged_df['hit_IDs'].fillna('[]')
merged_df['num_hits_x'] = merged_df['num_hits_x'].fillna(0)
merged_df['num_hits_y'] = merged_df['num_hits_y'].fillna(0)

#merged_df['true_track'] = merged_df.apply(lambda row: 1 if row['num_hits_y'] > 0 else 0, axis=1)

# Check the size of the DataFrame
print(f"Size of merged DataFrame: {merged_df.shape}")

print("Saving CSV...")
# Save the merged dataset to a new CSV file
output_file_path = f'ProcessedData/signal1_96_32652/evaluation_prep/merged_tracks_{command}.csv'
#merged_df.to_csv(output_file_path, index=False)

# Save the CSV with progress indicators
chunk_size = 10000  # Number of rows per chunk
num_chunks = len(merged_df) // chunk_size + 1

with open(output_file_path, 'w') as f:
    for i, chunk in enumerate(range(0, len(merged_df), chunk_size)):
        merged_df.iloc[chunk:chunk + chunk_size].to_csv(f, header=(i == 0), index=False)
        print(f"Saved chunk {i + 1} of {num_chunks}")


print(f"Merged CSV file saved to {output_file_path}")



"""
# Save the updated DataFrame to a new CSV file
output_file_path = f'ProcessedData/signal1_96_32652/evaluation_prep/predicted_tracks4_merged{command}.csv'
df.to_csv(output_file_path, index=False)

print(f"Updated CSV file saved to {output_file_path}")
"""