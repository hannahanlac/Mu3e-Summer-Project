import pandas as pd
import ast
import os

###################### Predicted Tracks - Condition 1 ##########################

# Load the CSV file
file_path = 'ProcessedData/signal1_95-99_32652/reconstructed_tracks/predicted_tracks5.csv'
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
command = '100/0'  # Change this to '100/100', '75/75', '50/50', etc. as needed
first_threshold, first_inclusive = threshold_mapping[command.split('/')[0]]
second_condition = command.split('/')[1]

print(f"Applying first truth threshold {command.split('/')[0]} with inclusivity: {first_inclusive}")

# Apply the function to each row and create new columns 'correct_track' and 'num_tid_hits'
df[['correct_track', 'num_tid_hits']] = df['tids'].apply(lambda x: check_correct_tracks(x, first_threshold, first_inclusive)).apply(pd.Series)

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
df['tid'] = df['tids'].apply(lambda x: update_tid_column(x, first_threshold, first_inclusive))

# Drop the original 'tids' column
df = df.drop(columns=['tids'])

####################### Test Data File #############################

# Load the test data parquet file and keep only the specified columns
test_data_file_path = 'ProcessedData/signal1_95-99_32652/test_data/test_data.parquet'
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


#################### Condition 2 ################

# Apply the second truth condition
def apply_second_condition(row, condition):
    if condition == '0':
        return row
    elif condition == '100':
        if row['num_hits_x'] != row['num_tid_hits']:
            row[['hit_IDs', 'num_hits_y', 'correct_track']] = [ [], 0, 0]
    elif condition == '75':
        if row['num_tid_hits'] / row['num_hits_x'] <= 0.75:
            row[['hit_IDs', 'num_hits_y', 'correct_track']] = [ [], 0, 0]
    elif condition == '50':
        if row['num_tid_hits'] / row['num_hits_x'] <= 0.50:
            row[['hit_IDs', 'num_hits_y', 'correct_track']] = [ [], 0, 0]
    return row

print(f"Applying second truth condition /{second_condition}")
merged_df = merged_df.apply(lambda row: apply_second_condition(row, second_condition), axis=1)



#################### Track Build Errors ################

# Toggle
overconstruction_search = True
duplicate_hit_search = True


"""
# If want to keep and label track build errors (complicated but kind :)
def track_build_errors(row):
    if overconstruction_search and row['num_tid_hits'] > row['num_hits_x']:
        row[['hit_IDs', 'num_hits_y', 'correct_track']] = [['overconstructed'], 0, 0]
    if duplicate_hit_search:
        hit_ids = ast.literal_eval(row['hit_IDs']) if isinstance(row['hit_IDs'], str) else row['hit_IDs']
        if len(hit_ids) != len(set(hit_ids)):
            row[['hit_IDs', 'num_hits_y', 'correct_track']] = [['hit duplicates'], 0, 0]
    return row

print("Applying track build errors...")
merged_df = merged_df.apply(track_build_errors, axis=1)
"""

# If want delete track build errors (simplest but harsh)
def track_build_errors(row):
    if overconstruction_search and row['num_tid_hits'] > row['num_hits_x']:
        return None  # Mark for deletion
    if duplicate_hit_search:
        hit_ids = ast.literal_eval(row['hit_IDs']) if isinstance(row['hit_IDs'], str) else row['hit_IDs']
        if len(hit_ids) != len(set(hit_ids)):
            return None  # Mark for deletion
    return row

print("Applying track build errors...")
merged_df = merged_df.apply(track_build_errors, axis=1).dropna()


###################### Managing TID Duplicates ####################################

# Toggleable option
tid_duplicates_option = 2  # Set to 1 for merging hit_IDs, 2 for keeping the highest num_hits_y

def tid_duplicates(df, option):
    if option == 1:
        # Merge hit_IDs for rows with the same tid
        df = df.groupby("tid").agg(
            frame_array=("frame_array", "first"), 
            traj_p=("traj_p", "first"), 
            traj_pt=("traj_pt", "first"), 
            traj_lambda=("traj_lambda", "first"), 
            traj_phi=("traj_phi", "first"), 
            num_hits_x=("num_hits_x", "first"),
            hit_IDs=("hit_IDs", lambda x: list(set(sum(x.dropna().apply(ast.literal_eval), [])))),  # Combine lists and remove duplicates
            num_hits_y=("num_hits_y", "sum"),
            num_tid_hits=("num_tid_hits", "sum"),
            correct_track=("correct_track", "first")
        ).reset_index()
    elif option == 2:
        # Keep the row with the highest num_hits_y for each tid
        df = df.sort_values('num_hits_y', ascending=False).drop_duplicates(subset=['tid'], keep='first').reset_index(drop=True)
    return df

print(f"Applying TID duplicates option {tid_duplicates_option}")
merged_df = tid_duplicates(merged_df, tid_duplicates_option)


##################### Save CSV ###########################

# Check the size of the DataFrame
print(f"Size of merged DataFrame: {merged_df.shape}")

print("Saving CSV...")
safe_command = str(command).replace("/", "-")

# Ensure the directory exists
output_directory = 'ProcessedData/signal1_95-99_32652/evaluation_prep'
os.makedirs(output_directory, exist_ok=True)

# Save the final DataFrame to a new CSV file
output_file_path = f'{output_directory}/predicted_tracks5_merged_{safe_command}_harsh.csv'
merged_df.to_csv(output_file_path, index=False)

print(f"Merged CSV file saved to {output_file_path}")