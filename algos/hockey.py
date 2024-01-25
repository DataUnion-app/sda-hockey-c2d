import json
import os
import sys
import shutil
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def get_job_details():
    """Reads in metadata information about assets used by the algo"""
    job = dict()
    job['dids'] = json.loads(os.getenv('DIDS', None))
    job['metadata'] = dict()
    job['files'] = dict()
    job['algo'] = dict()
    job['secret'] = os.getenv('secret', None)
    algo_did = os.getenv('TRANSFORMATION_DID', None)
    if job['dids'] is not None:
        for did in job['dids']:
            # get the ddo from disk
            filename = '/data/ddos/' + did
            print(f'Reading json from {filename}')
            with open(filename) as json_file:
                ddo = json.load(json_file)
                # search for metadata service
                for service in ddo['service']:
                    if service['type'] == 'metadata':
                        job['files'][did] = list()
                        index = 0
                        for file in service['attributes']['main']['files']:
                            job['files'][did].append(
                                '/data/inputs/' + did + '/' + str(index))
                            index = index + 1
    if algo_did is not None:
        job['algo']['did'] = algo_did
        job['algo']['ddo_path'] = '/data/ddos/' + algo_did
    return job


def get_input(local=False):
    if local:
        print("Reading local file")
        return "data/data.csv"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"/data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading file {filename}.")
        return filename


def prepare_for_visualization(player_id, strengths_dict, weaknesses_dict, avg_player, agg_df):
    print('preparing visualization')

    # filter dictionaries based on input variable
    player_strengths = strengths_dict[player_id]
    player_weaknesses = weaknesses_dict[player_id]
    # create dataframes
    strength_df = pd.DataFrame([player_strengths])
    weaknesses_df = pd.DataFrame([player_weaknesses])
    # insert Player ID into dataframes and reorder columns to make sure Player ID is the first columns
    strength_df['Player ID'] = player_id
    weaknesses_df['Player ID'] = player_id
    strength_df = strength_df[['Player ID'] + [col for col in strength_df.columns if col != 'Player ID']]
    weaknesses_df = weaknesses_df[['Player ID'] + [col for col in weaknesses_df.columns if col != 'Player ID']]
    # merge strengths and weaknesses together
    player_df = pd.merge(strength_df, weaknesses_df, on="Player ID", how="inner")
    # lookup position of player
    position = avg_player.loc[avg_player['Player ID'] == player_id, 'Position'].values[0]
    # get values of best player
    condition = avg_player['Position'] == position
    filtered_df = avg_player[condition]
    relevant_players = filtered_df["Player ID"].tolist()
    condition2 = agg_df["Player ID"].isin(relevant_players)
    filtered_df2 = agg_df[condition2]
    row_sums = filtered_df2.iloc[:, 1:].sum(axis=1)
    max_row_index = row_sums.idxmax()
    player_id_with_highest_sum = agg_df.loc[max_row_index, 'Player ID']
    bestval1 = agg_df.loc[agg_df['Player ID'] == player_id_with_highest_sum, player_df.columns[1]].values[0]
    bestval2 = agg_df.loc[agg_df['Player ID'] == player_id_with_highest_sum, player_df.columns[2]].values[0]
    bestval3 = agg_df.loc[agg_df['Player ID'] == player_id_with_highest_sum, player_df.columns[3]].values[0]
    bestval4 = agg_df.loc[agg_df['Player ID'] == player_id_with_highest_sum, player_df.columns[4]].values[0]
    bestval5 = agg_df.loc[agg_df['Player ID'] == player_id_with_highest_sum, player_df.columns[5]].values[0]
    bestval6 = agg_df.loc[agg_df['Player ID'] == player_id_with_highest_sum, player_df.columns[6]].values[0]
    best_player_row = pd.Series(["Best Player", bestval1, bestval2, bestval3, bestval4, bestval5, bestval6],
                                index=player_df.columns)
    player_df = pd.concat([player_df, pd.DataFrame([best_player_row])], ignore_index=True)
    # add row to dataframe with reference values (reference values are always 1)
    new_row = pd.Series(["avg. Position value", 1, 1, 1, 1, 1, 1], index=player_df.columns)
    player_df = pd.concat([player_df, pd.DataFrame([new_row])], ignore_index=True)

    return player_df


def visualization(player_df, output_dir):
    print('visualizing')
    BG_WHITE = "#fbf9f4"
    BLUE = "#2a475e"
    GREY70 = "#b3b3b3"
    GREY_LIGHT = "#f2efe8"
    COLORS = ["#FF5A5F", "#FFB400", "#007A87"]

    # define categories
    CATEGORIES = player_df["Player ID"].values.tolist()

    # define the six variables for the plot
    VARIABLES = player_df.columns.tolist()[1:]
    VARIABLES_N = len(VARIABLES)

    # define the angles at which the values of the numeric variables are placed
    ANGLES = [n / VARIABLES_N * 2 * np.pi for n in range(VARIABLES_N)]
    ANGLES += ANGLES[:1]

    # define padding used to customize the location of the tick labels
    X_VERTICAL_TICK_PADDING = 5
    X_HORIZONTAL_TICK_PADDING = 50

    # define angle values going from 0 to 2*pi
    HANGLES = np.linspace(0, 2 * np.pi)

    # Used for the equivalent of horizontal lines in cartesian coordinates plots
    # The last one is also used to add a fill which acts a background color.
    H0 = np.zeros(len(HANGLES))
    H1 = np.ones(len(HANGLES)) * 0.5
    H2 = np.ones(len(HANGLES))

    # Initialize layout ----------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, polar=True)

    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # Rotate the "" 0 degrees on top.
    # There it where the first variable, avg_bill_length, will go.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot lines and dots --------------------------------------------
    all_values = []
    for idx, species in enumerate(CATEGORIES):
        values = player_df.iloc[idx].drop("Player ID").values.tolist()
        values += values[:1]
        ax.plot(ANGLES, values, c=COLORS[idx], linewidth=4, label=species)
        ax.scatter(ANGLES, values, s=160, c=COLORS[idx], zorder=10)
        for value in values:
            all_values.append(value)

    all_values.sort()
    # setting limits for y axis
    ax.set_ylim(all_values[0] * 0.9, all_values[-1] * 1.1)

    # Set values for the angular axis (x)
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels(VARIABLES, size=14)

    # Remove lines for radial axis (y)
    ax.set_yticks([])

    # Remove spines
    ax.spines["start"].set_color("none")

    # add legends
    handles = [
        Line2D(
            [], [],
            c=color,
            lw=3,
            marker="o",
            markersize=8,
            label=species
        )
        for species, color in zip(CATEGORIES, COLORS)
    ]

    legend = ax.legend(
        handles=handles,
        loc=(1, 0),  # bottom-right
        labelspacing=1.5,  # add space between labels
        frameon=False  # don't put a frame
    )

    # Iterate through text elements and change their properties
    for text in legend.get_texts():
        text.set_fontsize(16)

        # Add title
    fig.suptitle(
        "Radar Plot of Top 3 Player strengths and weaknesses compared to the average and the best player of his Position group",
        x=0.1,
        y=1,
        ha="left",
        fontsize=14,
        color=BLUE,
        weight="bold",
    )

    fig.savefig(os.path.join(output_dir, "radarplot.png"), bbox_inches='tight')
    plt.close(fig)


def description(player_df, avg_player, output_dir):
    print('creating description')
    # get player id of current player
    player_id = player_df.at[0, "Player ID"]
    # get strengths and weaknesses (text, values and relative values)
    strengths1_text = player_df.columns[1]
    strengths1_value = "{:.2f}".format(
        avg_player.loc[avg_player['Player ID'] == player_id, player_df.columns[1]].values[0])
    strenghts1_relative = "{:.2f}".format(
        (player_df.loc[player_df['Player ID'] == player_id, player_df.columns[1]].values[0] - 1) * 100)
    strengths2_text = player_df.columns[2]
    strengths2_value = "{:.2f}".format(
        avg_player.loc[avg_player['Player ID'] == player_id, player_df.columns[2]].values[0])
    strenghts2_relative = "{:.2f}".format(
        (player_df.loc[player_df['Player ID'] == player_id, player_df.columns[2]].values[0] - 1) * 100)
    strengths3_text = player_df.columns[3]
    strengths3_value = "{:.2f}".format(
        avg_player.loc[avg_player['Player ID'] == player_id, player_df.columns[3]].values[0])
    strenghts3_relative = "{:.2f}".format(
        (player_df.loc[player_df['Player ID'] == player_id, player_df.columns[3]].values[0] - 1) * 100)
    weakness1_text = player_df.columns[4]
    weakness1_value = "{:.2f}".format(
        avg_player.loc[avg_player['Player ID'] == player_id, player_df.columns[4]].values[0])
    weakness1_relative = "{:.2f}".format(
        (1 - player_df.loc[player_df['Player ID'] == player_id, player_df.columns[4]].values[0]) * 100)
    weakness2_text = player_df.columns[5]
    weakness2_value = "{:.2f}".format(
        avg_player.loc[avg_player['Player ID'] == player_id, player_df.columns[5]].values[0])
    weakness2_relative = "{:.2f}".format(
        (1 - player_df.loc[player_df['Player ID'] == player_id, player_df.columns[5]].values[0]) * 100)
    weakness3_text = player_df.columns[6]
    weakness3_value = "{:.2f}".format(
        avg_player.loc[avg_player['Player ID'] == player_id, player_df.columns[6]].values[0])
    weakness3_relative = "{:.2f}".format(
        (1 - player_df.loc[player_df['Player ID'] == player_id, player_df.columns[6]].values[0]) * 100)
    # create string
    description = f"""
    Your three biggest strengths are:
    \t1. {strengths1_text} with an absolute value of {strengths1_value} which is {strenghts1_relative}% higher than the average of your Position group.
    \t2. {strengths2_text} with an absolute value of {strengths2_value} which is {strenghts2_relative}% higher than the average of your Position group.
    \t3. {strengths3_text} with an absolute value of {strengths3_value} which is {strenghts3_relative}% higher than the average of your Position group.\n
    Your three biggest weaknesses are:
    \t1. {weakness1_text} with an absolute value of {weakness1_value} which is {weakness1_relative}% lower than the average of your Position group.
    \t2. {weakness2_text} with an absolute value of {weakness2_value} which is {weakness2_relative}% lower than the average of your Position group.
    \t3. {weakness3_text} with an absolute value of {weakness3_value} which is {weakness3_relative}% lower than the average of your Position group.
    """

    # Specify the file path where you want to save the text document
    file_path = os.path.join(output_dir, "description.txt")
    # Open the file in write mode ('w')
    with open(file_path, 'w') as file:
        # Write the string data to the file
        file.write(description)


def strength_weakness(player_id, data, output_dir):
    print('analyzing overall strength and weakness for player id {}'.format(player_id))
    try:
        condition = data['Types'] == "Match"
        match_data = data[condition]
        match_data = match_data[
            ['Player ID', 'Position', 'Distance / min (m)', 'High Metabolic Power Distance / min (m)',
             'Acceleration Load (max.)', 'Speed (max.) (km/h)', 'Speed (Ø) (km/h)',
             'Acceleration (max.) (m/s²)', 'Deceleration (max.) (m/s²)', 'Accelerations / min',
             'Decelerations / min']]

        avg_player = match_data.groupby(['Player ID', 'Position'], as_index=False).mean()
        avg_position = match_data.groupby('Position', as_index=False).mean()
        avg_position = avg_position.drop("Player ID", axis=1)

        rows = avg_player.shape[0]
        values = []

        for i in range(rows):
            dict_values = {}
            pos = avg_player.at[i, "Position"]
            dict_values["Player ID"] = avg_player.at[i, "Player ID"]
            for col in avg_player.columns:
                if col != "Player ID" and col != "Position":
                    relative = avg_player.at[i, col] / avg_position.loc[avg_position['Position'] == pos, col].values[0]
                    dict_values[col] = relative
            values.append(dict_values)

        agg_df = pd.DataFrame(values)

        cols_for_evaluation = [col for col in agg_df.columns if col != "Player ID"]

        # Perform the nlargest calculation on the other columns
        strengths_df = (agg_df[cols_for_evaluation]
                        .stack()
                        .groupby(level=0)
                        .nlargest(3)
                        .unstack()
                        .reset_index(level=1, drop=True)
                        .reindex(columns=cols_for_evaluation))

        # Reinsert the "Player ID" column into strengths_df
        strengths_df.insert(0, 'Player ID', agg_df['Player ID'])

        # Perform the nlargest calculation on the other columns
        weaknesses_df = (agg_df[cols_for_evaluation]
                         .stack()
                         .groupby(level=0)
                         .nsmallest(3)
                         .unstack()
                         .reset_index(level=1, drop=True)
                         .reindex(columns=cols_for_evaluation))

        # Reinsert the "Player ID" column into strengths_df
        weaknesses_df.insert(0, 'Player ID', agg_df['Player ID'])

        rows = strengths_df.shape[0]

        strengths_dict = {}
        weaknesses_dict = {}

        for i in range(rows):
            strengths_dict[strengths_df.at[i, "Player ID"]] = {}
            for col in strengths_df.columns:
                if col != "Player ID" and not np.isnan(strengths_df.loc[i, col]):
                    strengths_dict[strengths_df.at[i, "Player ID"]][col] = strengths_df.at[i, col]

        for i in range(rows):
            weaknesses_dict[weaknesses_df.at[i, "Player ID"]] = {}
            for col in weaknesses_df.columns:
                if col != "Player ID" and not np.isnan(weaknesses_df.loc[i, col]):
                    weaknesses_dict[weaknesses_df.at[i, "Player ID"]][col] = weaknesses_df.at[i, col]

        player_df = prepare_for_visualization(player_id, strengths_dict, weaknesses_dict, avg_player, agg_df)
        visualization(player_df, output_dir)
        description(player_df, avg_player, output_dir)
    except Exception as e:
        print(e)


def per_period(player_id, session_id, data, output_dir):
    print('analyzing per period performance for player id {}'.format(player_id))
    try:
        condition = data['Types'] == "Period"
        match_data = data[condition]
        match_data = match_data[['Player ID', 'Description', 'Session ID', 'Position', 'Distance / min (m)',
                                 'High Metabolic Power Distance / min (m)', 'Acceleration Load (max.)',
                                 'Speed (max.) (km/h)', 'Speed (Ø) (km/h)', 'Acceleration (max.) (m/s²)',
                                 'Deceleration (max.) (m/s²)', 'Accelerations / min', 'Decelerations / min']]

        player_data = match_data
        avg_position = match_data.groupby(['Position', 'Session ID', 'Description'], as_index=False).mean()
        avg_position["Player ID"] = "avg of Position"

        # prepare visualization
        condition1 = player_data['Player ID'] == player_id
        condition2 = player_data['Session ID'] == session_id
        filtered_player_df = player_data[condition1 & condition2]
        position = player_data.loc[player_data['Player ID'] == player_id, 'Position'].values[0]
        condition_pos = avg_position["Position"] == position
        condition_pos2 = avg_position["Session ID"] == session_id
        filtered_pos_df = avg_position[condition_pos & condition_pos2]
        column_order = filtered_player_df.columns
        filtered_pos_df = filtered_pos_df[column_order]
        visualization_df = pd.concat([filtered_player_df, filtered_pos_df], ignore_index=True)

        # create subplot grid
        cols_not_needed = ["Player ID", "Description", "Session ID", "Position"]
        METRICS = [col for col in visualization_df.columns if col not in cols_not_needed]
        METRIC = METRICS[0]
        PLAYERS = visualization_df["Player ID"].unique()
        PLAYER = PLAYERS[0]
        # Get the metric columns
        metric_columns = METRICS

        # Get unique IDs from the 'Player ID' column
        unique_ids = visualization_df['Player ID'].unique()

        # Create a subplot grid based on the number of metric columns
        num_metrics = len(metric_columns)
        num_rows = (num_metrics + 2) // 3
        num_cols = min(num_metrics, 3)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

        for idx, metric_col in enumerate(metric_columns):
            row = idx // num_cols
            col = idx % num_cols

            for unique_id in unique_ids:
                id_filter = visualization_df['Player ID'] == unique_id
                data = visualization_df[id_filter][['Description', metric_col]]

                ax = axes[row, col] if num_metrics > 1 else axes
                if unique_id == PLAYER:
                    # Highlight points for ID = selected player with dots and labels
                    ax.plot(data['Description'], data[metric_col], label=f'{unique_id}', color="#0b53c1", lw=2.4,
                            zorder=10)
                    ax.scatter(data['Description'], data[metric_col], fc="w", ec="#0b53c1", s=60, lw=2.4, zorder=12)
                else:
                    # Plot other IDs normally
                    ax.plot(data['Description'], data[metric_col], label=f'{unique_id}', color="#BFBFBF", lw=1.5)

                ax.set_title(metric_col)
                ax.set_xlabel('Description')
                ax.set_ylabel('Value')
                ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'linechart.png'))
        plt.close()
    except Exception as e:
        print(e)

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

def main(job_details, local=False):
    print('Starting compute job with the following input information:')
    if job_details is not None:
        print(json.dumps(job_details, sort_keys=True, indent=4))

    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return

    output_dir = "./outputs/result" if local else "/data/outputs/result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fields = ['Player ID', 'Position', 'Distance / min (m)', 'High Metabolic Power Distance / min (m)',
             'Acceleration Load (max.)', 'Speed (max.) (km/h)', 'Speed (Ø) (km/h)',
             'Acceleration (max.) (m/s²)', 'Deceleration (max.) (m/s²)', 'Accelerations / min',
             'Decelerations / min', 'Types', 'Description', 'Session ID']
    frames = []
    try:
        with zipfile.ZipFile(filename) as zf:
            for file in zf.namelist():
                print(f"Reading file in zip [{file}].")
                if file != "data.json":
                    try:
                        with zf.open(file) as f:
                            file_data = pd.read_csv(f, sep=";", encoding_errors="ignore", skip_blank_lines=True, usecols=fields).dropna(how='all')
                            frames.append(file_data)
                    except Exception as e:
                        print(str(e))
                        pass
            data = pd.concat(frames)
            print(data['Player ID'].unique())
            for player_id in data['Player ID'].unique():
                player_dir = "{}/{}".format(output_dir, player_id)

                if not os.path.exists(player_dir):
                    os.makedirs(player_dir)

                strength_weakness(player_id, data, player_dir)
                per_period(player_id, 234, data, player_dir)

    except zipfile.BadZipFile as error:
        print("Error reading zipfile [%s]", file, exc_info=error)
        sys.exit(-1)

    output_filename = "./outputs/result.zip" if local else "/data/outputs/result.zip"
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(output_dir, zipf)

    shutil.rmtree(output_dir)

    print('Done!')


if __name__ == '__main__':
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    main(None, local)
