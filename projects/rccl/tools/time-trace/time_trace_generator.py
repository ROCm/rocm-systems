import os
import random
import pandas as pd
import plotly.graph_objects as go
import argparse

def generateRandomColors(df, colorList):

    for _ in range(len(df)):
        r = random.random()
        g = random.random()
        b = random.random()
        colorList.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")
    
    return colorList

def plotCompileTime(log_file, minVal):
    colors = []

    # read the log file and extract the data from it
    # st:   start-time (ms)
    # et:   end-time (ms)
    # ts:   timestamp
    # file: path to file
    # hash: command hash 
    df = pd.read_csv(log_file, delimiter='\t', header=None, 
                     names=['st', 'et', 'ts', 'file', 'hash'])
    df = df.iloc[1:]

    # include file name only
    df['file'] = df['file'].apply(os.path.basename)

    # convert to seconds
    df['st'] = df['st'].astype(int) / 1000
    df['et'] = df['et'].astype(int) / 1000

    # calculate compilation duration of the file
    df['dur'] = df['et'].astype(int) - df['st'].astype(int)

    if args.include_linking == 0:
        # drop the last two rows which are related to linking
        df = df.drop(df.index[-2:]) 
    
    # if minVal specified remove the rows from the df where df['dur'] < minVal
    df = df[df['dur'] >= minVal]

    maxEt = int(df['et'].max())
    df = df[::-1]   # reverse df

    colors = generateRandomColors(df, colors)

    fig = go.Figure(go.Bar(
        y=df['file'],
        x=df['dur'],
        orientation='h',
        marker=dict(color=colors),
        base=df['st'],
        textposition='auto',
        customdata=df['dur'],
        hovertemplate='<b>Time:</b> %{customdata} seconds<br>' +
                    '<b>File Name:</b> %{y}<br>'
    ))

    # Customize the layout
    fig.update_layout(
        title="RCCL TOTAL COMPILE TIME LINE",
        xaxis_title='Duration (seconds)',
        yaxis_title='file name',
        bargap=0.1,
        plot_bgcolor='#36454F',  # Set the plot background color to black
        paper_bgcolor='#36454F',  # Set the paper background color to black
        font=dict(
            family="Arial",
            size=11,
            color="white"
        ),
    )

    # add custom text annotation at the top right corner
    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                x=1,
                y=1,
                xref="paper",
                yref="paper",
                text="Total Time: "+ str(maxEt) + " seconds",
                showarrow=False,
                font=dict(
                    size=18,
                    color="white"
                )
            )
        ]
    )

    # convert the plot to an html file
    fig.write_html("RCCL-compile-timeline.html", auto_open=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_val", nargs='?', default='5', type=int, help="Ignore any if it's less than the value provided")
    parser.add_argument("--include_linking", action='store_true', help="Include linking when plotting")
    parser.add_argument("--log_file_path", type=str, help="Location of the log file generated with --time-trace flag")
    args = parser.parse_args()

    if args.log_file_path is not None:
        log_file_path = args.log_file_path
    else:
        log_file_path = '../../build/release/time_trace.log'

    plotCompileTime(log_file_path, args.min_val)