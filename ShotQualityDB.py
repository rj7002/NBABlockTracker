import re
import pandas as pd
import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ast
import nba_api.stats.static.teams as nba
import unicodedata
from datetime import datetime, timedelta
import nba_api.stats.endpoints.playbyplayv3 as pbp
from zipfile import ZipFile

st.set_page_config(page_title='NBA Block Tracker',layout='wide',page_icon='https://www.shutterstock.com/image-vector/vector-illustration-basket-basketball-game-600nw-2417843359.jpg')
st.markdown(
    """
    <h1 style='text-align: center; 
               font-size: 64px; 
               color: #39ff14; 
               text-shadow: 0 0 5px #39ff14, 
                            0 0 10px #39ff14, 
                            0 0 20px #39ff14, 
                            0 0 40px #0fa, 
                            0 0 80px #0fa;'>
        NBA Block Tracker
    </h1>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <h1 style='text-align: center; 
               font-size: 40px; 
               color: #39ff14; 
               text-shadow: 0 0 5px #39ff14, 
                            0 0 10px #39ff14, 
                            0 0 20px #39ff14, 
                            0 0 40px #0fa, 
                            0 0 80px #0fa;'>
        NBA Block Tracker
    </h1>
    """,
    unsafe_allow_html=True
)
def parse_time_str(time_str):
                """Convert a string like '1:19.0' into a timedelta object."""
                try:
                    minutes, seconds = time_str.split(':')
                    return timedelta(minutes=int(minutes), seconds=float(seconds))
                except Exception as e:
                    print("Error parsing time:", e)
                    return None
def fixTime(iso_str):
                # Match minutes and seconds using regex
                match = re.match(r"PT(\d+)M(\d+(?:\.\d+)?)S", iso_str)
                if not match:
                    return None  # or raise an error
                
                minutes, seconds = match.groups()

                # If seconds ends in .0, strip it
                seconds = str(float(seconds)).rstrip('0').rstrip('.') 
                
                return f"{int(minutes)}:{seconds}"


def draw_plotly_court(fig, fig_width=500, fig_height=870, margins=10, lwidth=3,
                    show_title=True, labelticks=True, show_axis=True,
                    glayer='below', bg_color='white'):
    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5,
                    start_angle=0.0, end_angle=2 * np.pi, N=200,
                    closed=False, opposite=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        if opposite:
            y = y_center + b * np.sin(-t)
        else:
            y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    ####################################################################
    ############################ dimensions ############################
    #  half-court -52.5 <= y <= 417.5, full-court -52.5 <= y <= 887.5  #
    ####################################################################
    fig.update_layout(height=870,
                    template='plotly_dark')

    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins],
                    visible=show_title)
    fig.update_yaxes(range=[-52.5 - margins, 887.5 + margins],
                    visible=show_title)

    threept_break_y = 89.47765084
    # three_line_col = "#000000"
    # main_line_col = "#000000"
    three_line_col = "white"
    main_line_col = "white"

    fig.update_layout(
        # Line Horizontal
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            fixedrange=True,
            visible=show_axis,
            showticklabels=labelticks,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            fixedrange=True,
            visible=show_axis,
            showticklabels=labelticks,
        ),
        yaxis2=dict(
            scaleanchor="x2",
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            fixedrange=True,
            visible=show_axis,
            showticklabels=labelticks,
            range=[-52.5 - margins, 887.5 + margins]
        ),
        xaxis2=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            fixedrange=True,
            visible=show_axis,
            showticklabels=labelticks,
            range=[-250 - margins, 250 + margins]
        ),
        showlegend=False,
        shapes=[
            # court border
            dict(
                type="rect", x0=-250, y0=-52.5, x1=250, y1=887.5,
                line=dict(color=main_line_col, width=lwidth),
                # fillcolor='#333333',
                layer=glayer
            ),
            # half-court line
            dict(
                type="line", x0=-250, y0=417.5, x1=250, y1=417.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            # back-court outer ft-lines
            dict(
                type="line", x0=-80, y0=697.5, x1=-80, y1=887.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            dict(
                type="line", x0=80, y0=697.5, x1=80, y1=887.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            # back-court inner ft-lines
            dict(
                type="line", x0=-60, y0=697.5, x1=-60, y1=887.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            dict(
                type="line", x0=60, y0=697.5, x1=60, y1=887.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            # front-court outer ft-lines
            dict(
                type="line", x0=-80, y0=-52.5, x1=-80, y1=137.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            dict(
                type="line", x0=80, y0=-52.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            # front-court inner ft-lines
            dict(
                type="line", x0=-60, y0=-52.5, x1=-60, y1=137.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            dict(
                type="line", x0=60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            # back-court ft-circle
            dict(
                type="circle", x0=-60, y0=637.5, x1=60, y1=757.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=lwidth),
                # fillcolor='#dddddd',
                layer=glayer
            ),
            # front-court ft-circle
            dict(
                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=lwidth),
                # fillcolor='#dddddd',
                layer=glayer
            ),
            # back-court ft-line
            dict(
                type="line", x0=-80, y0=697.5, x1=80, y1=697.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            # front-court ft-line
            dict(
                type="line", x0=-80, y0=137.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=lwidth),
                layer=glayer
            ),
            # back-court basket
            dict(
                type="circle", x0=-7.5, y0=827.5, x1=7.5, y1=842.5, xref="x", yref="y",
                line=dict(color="#ec7607", width=lwidth),
            ),
            dict(
                type="rect", x0=-2, y0=842.25, x1=2, y1=847.25,
                line=dict(color="#ec7607", width=lwidth),
                fillcolor='#ec7607',
            ),
            dict(
                type="line", x0=-30, y0=847.5, x1=30, y1=847.5,
                line=dict(color="#ec7607", width=lwidth),
            ),
            # front-court basket
            dict(
                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                line=dict(color="#ec7607", width=lwidth),
            ),
            dict(
                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                line=dict(color="#ec7607", width=lwidth),
                fillcolor='#ec7607',
            ),
            dict(
                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color="#ec7607", width=lwidth),
            ),
            # back-court charge circle
            dict(type="path",
                path=ellipse_arc(y_center=835, a=40, b=40,
                                start_angle=0, end_angle=np.pi, opposite=True),
                line=dict(color=main_line_col, width=lwidth), layer=glayer),
            # front-court charge circle
            dict(type="path",
                path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                line=dict(color=main_line_col, width=lwidth), layer=glayer),
            # back-court 3pt line
            dict(type="path",
                path=ellipse_arc(y_center=835, a=237.5, b=237.5, start_angle=np.pi - \
                                                                            0.386283101, end_angle=0.386283101,
                                opposite=True),
                line=dict(color=main_line_col, width=lwidth), layer=glayer),
            # front-court 3pt line
            dict(type="path",
                path=ellipse_arc(
                    a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                line=dict(color=main_line_col, width=lwidth), layer=glayer),
            # back-court corner three lines
            dict(
                type="line", x0=-220, y0=835 - threept_break_y, x1=-220, y1=887.5,
                line=dict(color=three_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=220, y0=835 - threept_break_y, x1=220, y1=887.5,
                line=dict(color=three_line_col, width=lwidth), layer=glayer
            ),
            # front-court corner three lines
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                line=dict(color=three_line_col, width=lwidth), layer=glayer
            ),
            # back-court tick marks
            dict(
                type="line", x0=-250, y0=607.5, x1=-220, y1=607.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=250, y0=607.5, x1=220, y1=607.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            # front-court tick marks
            dict(
                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=250, y0=227.5, x1=220, y1=227.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            # front-court free-throw tick marks
            dict(
                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=90, y0=17.5, x1=80, y1=17.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=90, y0=27.5, x1=80, y1=27.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=90, y0=57.5, x1=80, y1=57.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=90, y0=87.5, x1=80, y1=87.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            # back-court free-throw tick marks
            dict(
                type="line", x0=-90, y0=817.5, x1=-80, y1=817.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=-90, y0=807.5, x1=-80, y1=807.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=-90, y0=777.5, x1=-80, y1=777.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=-90, y0=747.5, x1=-80, y1=747.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=90, y0=817.5, x1=80, y1=817.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=90, y0=807.5, x1=80, y1=807.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=90, y0=777.5, x1=80, y1=777.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            dict(
                type="line", x0=90, y0=747.5, x1=80, y1=747.5,
                line=dict(color=main_line_col, width=lwidth), layer=glayer
            ),
            # half-court outer circle
            dict(type="path",
                path=ellipse_arc(y_center=417.5, a=60, b=60,
                                start_angle=0, end_angle=2 * np.pi),
                line=dict(color=main_line_col, width=lwidth), layer=glayer),
            # half-court inner circle
            dict(type="path",
                path=ellipse_arc(y_center=417.5, a=25, b=25,
                                start_angle=0, end_angle=2 * np.pi),
                line=dict(color=main_line_col, width=lwidth), layer=glayer),
        ]
    )
    return True

def display_player_image(player_id, width2, caption2):
    # Construct the URL for the player image using the player ID
    image_url = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
    
    # Check if the image URL returns a successful response
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{caption2}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
        # st.image(image_url, width=width2, caption=caption2)
    else:
        image_url = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{"Image Unavailable"}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

def display_player_image2(player_id, width2, caption2):
    # Construct the URL for the player image using the player ID
    image_url = f"https://cdn.nba.com/logos/nba/{player_id}/primary/D/logo.svg"
    
    # Check if the image URL returns a successful response
    response = requests.head(image_url)
    
    if response.status_code == 200:
        # If image is available, display it
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{caption2}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
        # st.image(image_url, width=width2, caption=caption2)
    else:
        image_url = "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"
        st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center;">'
        f'<img src="{image_url}" style="width: {width2}px;">'
        f'<p style="text-align: center;">{"Image Unavailable"}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

season = st.selectbox('Select Season',['2021-22','2022-23','2023-24','2024-25'])
zip_path = f'{season}_blocks.csv.zip'

with ZipFile(zip_path) as z:
    # Pick the first CSV that does NOT start with __MACOSX
    csv_file = [f for f in z.namelist() if not f.startswith('__MACOSX')][0]
    fulldf = pd.read_csv(z.open(csv_file))
if season != '2024-25':
    fulldf['block_team_name'] = np.where(
        fulldf['possession_team'] == fulldf['home_team_name'],
        fulldf['away_team_name'],   # opposite of possession = away when possession is home
        fulldf['home_team_name']   # otherwise opposite = home when possession is away
    )
    teamorplayer = st.pills('Filter by team or player',['Team','Player'],selection_mode='single',default='Player')
    if teamorplayer == 'Team':
        team_options = fulldf['block_team_name'].unique()
        team = st.selectbox('Select a team', team_options)
        nbateamid = nba.find_teams_by_full_name(team)[0]["id"]
    
    else:
        import nba_api.stats.endpoints.playerindex as nba
        df = nba.PlayerIndex(season='2023-24',historical_nullable=1,active_nullable=1).get_data_frames()[0]
        df['fullName'] = df['PLAYER_FIRST_NAME'] + " " + df['PLAYER_LAST_NAME']
        df['teamName'] = df['TEAM_CITY'] + " " + df['TEAM_NAME']
        players = df['fullName'].unique()
        player = st.selectbox('Select a player',players)
        df = df[df['fullName'] == player]
        playerid = df['PERSON_ID'].iloc[0]
        team = df['teamName'].iloc[0]
        nbateamid = df['TEAM_ID'].iloc[0]

    
        player = unicodedata.normalize('NFKD', player).encode('ASCII', 'ignore').decode('utf-8')
else:
    import nba_api.stats.endpoints.playerindex as nba
    df = nba.PlayerIndex(season='2023-24',historical_nullable=1,active_nullable=1).get_data_frames()[0]
    df['fullName'] = df['PLAYER_FIRST_NAME'] + " " + df['PLAYER_LAST_NAME']
    df['teamName'] = df['TEAM_CITY'] + " " + df['TEAM_NAME']
    players = df['fullName'].unique()
    player = st.selectbox('Select a player',players)
    df = df[df['fullName'] == player]
    playerid = df['PERSON_ID'].iloc[0]
    team = df['teamName'].iloc[0]
    nbateamid = df['TEAM_ID'].iloc[0]

from nba_api.stats.endpoints import leaguegamefinder

gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=nbateamid)
gamefind = gamefinder.get_data_frames()[0]

# types = pd.DataFrame(fulldf.dtypes.reset_index())
# types.columns = ['index', 'dtype']
# types['name'] = types['index'] + " - " + types["dtype"].astype(str)
# for index, row in types.iterrows():
#     st.write(row['name'])
if season != '2024-25':
    if teamorplayer == 'Player':
        display_player_image(playerid,300,'')
        fulldf = fulldf[fulldf['secondary_name'] == player]
    else:
        display_player_image2(nbateamid,300,'')
        fulldf = fulldf[fulldf['block_team_name'] == team]
else:
    display_player_image(playerid,300,'')
    fulldf = fulldf[fulldf['secondary_name'] == player]

st.warning('Note: Some blocks may be missing due to missing tracking data')
if season != '2024-25':
    hover_template = (
        "<b>Blocker<b>: %{customdata[0]}<br>" +
        "<b>Shooter<b>: %{customdata[1]}<br>" + 
        "<b>Distance Covered<b>: %{customdata[2]} ft<br>" + 
        "<b>Block Angle<b>: %{customdata[9]}°<br>" +
        "<b>Period<b>: %{customdata[3]}<br>" +
        "<b>Time<b>: %{customdata[4]}<br>" + 
        "<b>Shot Type<b>: %{customdata[5]} (%{customdata[6]})<br>" + 
        "<b>Game<b>: %{customdata[7]}<br>" + 
        "<b>Date<b>: %{customdata[8]}<br>" 

    )
else:
    hover_template = (
        "<b>Blocker<b>: %{customdata[0]}<br>" +
        "<b>Shooter<b>: %{customdata[1]}<br>" + 
        "<b>Distance Covered<b>: %{customdata[2]} ft<br>" + 
        "<b>Block Angle<b>: %{customdata[7]}°<br>" +
        "<b>Period<b>: %{customdata[3]}<br>" +
        "<b>Time<b>: %{customdata[4]}<br>" + 
        "<b>Shot Type<b>: %{customdata[5]} (%{customdata[6]})<br>" 
    )
block = fulldf
# teamids = block['team_id'].unique()
# block = block[(block['court_player_name'] == block['secondary_name'])]
# block['team_id'] = block['team_id'].astype(int)
# teamid = int(teamid)
# block = block[block['team_id'] != teamid]
block['distance_covered'] = round(np.sqrt((block['shot_x'] - block['court_x'])**2 + (block['shot_y'] - block['court_y'])**2),2)
block['play_descriptors'] = block['play_descriptors'].apply(ast.literal_eval)

# block['angle_degrees'] = round(np.degrees(block['Angle with the Closest Defender']))
# block['angle_degrees'] = block['angle_degrees'].apply(lambda x: x - 90 if x > 90 else x)

block['court_x'] = 10*block['court_x']-50
block['court_y'] = 10*block['court_y']-250
block['shot_x'] = 10*block['shot_x']-50
block['shot_y'] = 10*block['shot_y']-250

block['time'] = block['minutes'].astype(str) + ':' + block['seconds'].astype(str)
# block['feature_store'] = block['feature_store'].apply(json.loads)
# feature_df = block['feature_store'].apply(pd.Series)
# block = block.drop(columns=['feature_store'])  # Optional
# block = pd.concat([block, feature_df], axis=1)
block['ClosestDefVelBlockAng'] = round(np.rad2deg(block['Closest Defender Velocity Angle']))
st.sidebar.title('Filters')
blockers = block['secondary_name'].unique()
shooters = block['shooter_name'].unique()
blocker = st.sidebar.multiselect('Select blocker',blockers)
shooter = st.sidebar.multiselect('Select shooters',shooters)
period_filter = st.sidebar.selectbox(
        "Period", 
        options=["All"] + sorted(block['period'].unique().tolist())
    )

minutesmin, minutesmax = st.sidebar.slider(
        "Minutes", 
        min_value=int(block['minutes'].min()), 
        max_value=int(block['minutes'].max()), 
        value=(int(block['minutes'].min()), int(block['minutes'].max()))
    )
block = block[
        (block['minutes'] >= minutesmin) & 
        (block['minutes'] <= minutesmax)
    ]
if period_filter != "All":
        block = block[block['period'] == period_filter]
unique_descriptors = set([descriptor for sublist in block['play_descriptors'] for descriptor in sublist])
unique_descriptors = sorted(unique_descriptors)
selected_descriptors = st.sidebar.multiselect("Choose play descriptors", unique_descriptors)
if selected_descriptors:
    block = block[block['play_descriptors'].apply(lambda x: any(descriptor in x for descriptor in selected_descriptors))]
if shooter:
    block = block[block['shooter_name'].isin(shooter)]
if blocker:
    block = block[block['secondary_name'].isin(blocker)]
dist_min, dist_max = st.sidebar.slider(
        "Distance Covered (ft)", 
        min_value=int(block['distance_covered'].min()), 
        max_value=int(block['distance_covered'].max()), 
        value=(int(block['distance_covered'].min()), int(block['distance_covered'].max()))
    )
block = block[
        (block['distance_covered'] >= dist_min) & 
        (block['distance_covered'] <= dist_max)
    ]
# st.write(len(block))
arrows = st.sidebar.checkbox('Hide arrows',value=False)
# location = st.sidebar.radio('Block Location',['Start','End','None'])

court_fig = go.Figure()
draw_plotly_court(court_fig, show_title=False, labelticks=False, show_axis=False,
                            glayer='above', bg_color='dark gray', margins=0)
if arrows == False:
    for i in range(len(block)):
        court_fig.add_trace(go.Scatter(
                    x=[block['court_y'].iloc[i], block['shot_y'].iloc[i]],  # Passer and Shooter x coordinates
                    y=[block['court_x'].iloc[i], block['shot_x'].iloc[i]],  # Passer and Shooter y coordinates
                    mode='lines+markers',
                    line=dict(color='gold', width=2, shape='linear'),  # Arrow properties
                    marker=dict(size=10, color='gold',symbol='arrow',angleref="previous",opacity=1),  # Optional: Add a marker at the passer's location
                    opacity=0.5,
                    # customdata=passer[['court_player_name','shooter_name']].iloc[i],  # Use customdata for makes only
                    hoverinfo='none',  # Set hoverinfo to text
                    # hovertemplate=hover_template
                    # name=f'Pass to Shooter {i+1}',
                ))
# if location == 'Start':
#     court_fig.add_trace(go.Scatter(
#             x=block['court_y'], 
#             y=block['court_x'], 
#             mode='markers',
#             marker=dict(size=10, color='blue', opacity=1,symbol='circle'),
#             name='',
#             customdata=block[['court_player_name','shooter_name','distance_covered','period','time','action_type','play_descriptors','game_name','game_datetime_utc']],  # Use customdata for makes only
#             hoverinfo='text',  # Set hoverinfo to text
#             hovertemplate=hover_template
#         ))
# if location == 'End':
# block.loc[block['court_x'] > 425, 'court_x'] = 1000 - block.loc[block['court_x'] > 425, 'court_x']
if season != '2024-25':
        cdata = block[['court_player_name','shooter_name','distance_covered','period','time','action_type','play_descriptors','game_name','game_datetime_utc','ClosestDefVelBlockAng']]
else:
        cdata = block[['court_player_name','shooter_name','distance_covered','period','time','action_type','play_descriptors','ClosestDefVelBlockAng']]
court_fig.add_trace(go.Scatter(
        x=block['shot_y'], 
        y=block['shot_x'], 
        mode='markers',
        marker=dict(size=10, color='red', opacity=1,symbol='x'),
        name='',
        customdata=cdata,  # Use customdata for makes only
        hoverinfo='text',  # Set hoverinfo to text
        hovertemplate=hover_template
    ))
c1,c2 = st.columns(2)
with c1:
    # st.plotly_chart(court_fig,use_container_width=False)
    eventdata = st.plotly_chart(court_fig,use_container_width=True,on_select='rerun')
block1 = block[block['court_x'] > 425]
block2 = block[block['court_x'] <= 425]
# st.write(block1)

court_fig2 = go.Figure()
draw_plotly_court(court_fig2, show_title=False, labelticks=False, show_axis=False,
                            glayer='above', bg_color='dark gray', margins=0)
# if location == 'Start':
#     court_fig2.add_trace(go.Histogram2dContour(
#         x=block1['court_y'],  # Passer x-coordinates
#         y=block1['court_x'],  # Passer y-coordinates
#         # colorscale=[[0, 'black'], [0.1, 'black'],[0.3, 'yellow'], [1, 'red']],  # Color scale for the contour
#         colorscale=[[0, 'black'],[0.2, 'yellow'], [1, 'red']],  # Color scale for the contour
#         # opacity=0.6,  # Make the contour semi-transparent
#         contours=dict(
#             coloring='heatmap',  # Coloring the contours based on heatmap density
#             showlabels=False,  # Show contour labels (optional)
#             labelfont=dict(size=12)  # Font size of the contour labels (optional)
#         ),
#         showlegend=False
#     ))
#     court_fig2.add_trace(go.Histogram2dContour(
#         x=block2['court_y'],  # Passer x-coordinates
#         y=block2['court_x'],  # Passer y-coordinates
#         # colorscale=[[0, 'black'], [0.1, 'black'],[0.3, 'yellow'], [1, 'red']],  # Color scale for the contour
#         colorscale=[[0, 'black'],[0.2, 'yellow'], [1, 'red']],  # Color scale for the contour
#         # opacity=0.6,  # Make the contour semi-transparent
#         contours=dict(
#             coloring='heatmap',  # Coloring the contours based on heatmap density
#             showlabels=False,  # Show contour labels (optional)
#             labelfont=dict(size=12)  # Font size of the contour labels (optional)
#         ),
#         showlegend=False
#     ))
# if location == 'End':
court_fig2.add_trace(go.Histogram2dContour(
x=block1['shot_y'],  # Passer x-coordinates
y=block1['shot_x'],  # Passer y-coordinates
# colorscale=[[0, 'black'], [0.1, 'black'],[0.3, 'yellow'], [1, 'red']],  # Color scale for the contour
colorscale=[[0, 'black'],[0.2, 'yellow'], [1, 'red']],  # Color scale for the contour
# opacity=0.6,  # Make the contour semi-transparent
contours=dict(
    coloring='heatmap',  # Coloring the contours based on heatmap density
    showlabels=False,  # Show contour labels (optional)
    labelfont=dict(size=12)  # Font size of the contour labels (optional)
),
showlegend=False,
hovertemplate='Blocks: %{z}<extra></extra>'

))
court_fig2.add_trace(go.Histogram2dContour(
    x=block2['shot_y'],  # Passer x-coordinates
    y=block2['shot_x'],  # Passer y-coordinates
    # colorscale=[[0, 'black'], [0.1, 'black'],[0.3, 'yellow'], [1, 'red']],  # Color scale for the contour
    colorscale=[[0, 'black'],[0.2, 'yellow'], [1, 'red']],  # Color scale for the contour
    # opacity=0.6,  # Make the contour semi-transparent
    contours=dict(
        coloring='heatmap',  # Coloring the contours based on heatmap density
        showlabels=False,  # Show contour labels (optional)
        labelfont=dict(size=12)  # Font size of the contour labels (optional)
    ),
    showlegend=False,
    hovertemplate='Blocks: %{z}<extra></extra>'
))
court_fig2.update_traces(showscale=False)
with c2:
    st.plotly_chart(court_fig2,use_container_width=False)
if eventdata and "selection" in eventdata and eventdata["selection"]["points"]:
    x = eventdata["selection"]["points"][0]["x"]
    y = eventdata["selection"]["points"][0]["y"]
    date = eventdata["selection"]["points"][0]["customdata"][8]
    time = eventdata["selection"]["points"][0]["customdata"][4]
    time = time.replace(".0", "")
    period = eventdata["selection"]["points"][0]["customdata"][3]
    date_obj = datetime.strptime(date, "%m-%d-%Y")
    for offset in [0, -1, 1]:
        current_date = date_obj + timedelta(days=offset)
        formatted_date = current_date.strftime("%Y-%m-%d")
        
        filtered_games = gamefind[gamefind["GAME_DATE"] == formatted_date]
        
        if not filtered_games.empty:
            # st.write(f"Using date: {formatted_date}")
            gamefind = filtered_games
            break
    nbagameid = gamefind["GAME_ID"].iloc[0]
    oldgameid = nbagameid
    # nbagameid = str(nbagameid)[2:]
    # st.write(gamefind)
    # st.write(x)
    # st.write(y)
    # st.write(nbagameid)
    pbp = pbp.PlayByPlayV3(game_id=oldgameid).get_data_frames()[0]
    
    pbp['time'] = pbp['clock'].apply(fixTime)
    # st.write(pbp)

    # Convert your input time string to timedelta
    target_time = parse_time_str(time)

    # First try exact match
    for offset in [0, -1, 1]:  # seconds
        adjusted_time = target_time + timedelta(seconds=offset)
        
        # Convert back to string to match the format in your DataFrame
        minutes = int(adjusted_time.total_seconds() // 60)
        seconds = adjusted_time.total_seconds() % 60
        adjusted_time_str = f"{minutes}:{seconds:.1f}".rstrip('0').rstrip('.')

        # Apply filter
        filtered_pbp = pbp[(pbp['period'] == period) & (pbp['time'] == adjusted_time_str)]

        if not filtered_pbp.empty:
            pbp = filtered_pbp
            break
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Connection': 'keep-alive',
        'Referer': 'https://stats.nba.com/',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache'
    }
    actionid = pbp["actionNumber"].iloc[0]
    # url = f'https://api.databallr.com/api/get_video/{nbagameid}/{actionid}'
    url = 'https://stats.nba.com/stats/videoeventsasset?GameEventID={}&GameID={}'.format(
                    actionid, nbagameid)
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        json = r.json()
        video_urls = json['resultSets']['Meta']['videoUrls']
        playlist = json['resultSets']['playlist']
        video_event = {'video': video_urls[0]['lurl'], 'desc': playlist[0]['dsc']}
        video = video_urls[0]['lurl']
        # video_url = response.text.strip()
        # st.markdown(
        #     f"""
        #     <div style="text-align: center;">
        #         <video controls autoplay width="500" style="margin: auto;">
        #             <source src="{video_url}" type="video/mp4">
        #             Your browser does not support the video tag.
        #         </video>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
        st.video(video,autoplay=True)
    else:
        st.error('ERROR')
ca1,ca2 = st.columns(2)
with ca1:
    histfig = px.histogram(data_frame=block,x='distance_covered')
    histfig.update_traces(marker_line_color='black', marker_line_width=1)
    histfig.update_layout(
    title="Distance Covered Distribution",
    xaxis_title="Distance Covered (ft)",
    yaxis_title="Count"
)
    st.plotly_chart(histfig)
with ca2:
    barfig = px.histogram(block,x='period',color='action_type')
    barfig.update_traces(marker_line_color='black', marker_line_width=1)
    barfig.update_layout(title='Block Counts by Period')
    st.plotly_chart(barfig)
cb1,cb2 = st.columns(2)
with cb1:
    block2 = block
    # passer2['play_descriptors'] = passer2['play_descriptors'].apply(ast.literal_eval)
    block_exploded = block2.explode('play_descriptors')
    fig = px.box(block_exploded, 
            x='play_descriptors', 
            y='distance_covered', 
            title='Distance Covered Distribution by Descriptor',
            points='outliers')

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
with cb2:
    if blocker or teamorplayer:
        counts = block['shooter_name'].value_counts().reset_index().head(10)
        counts.columns = ['shooter_name', 'count']
        piefig = px.pie(counts, names='shooter_name', values='count', title='Top 10 Players Blocked')
    else:
        counts = block['secondary_name'].value_counts().reset_index()
        counts.columns = ['secondary_name', 'count']
        piefig = px.pie(counts, names='secondary_name', values='count', title='Block Distribution by Player')
    st.plotly_chart(piefig)
st.sidebar.subheader('')
st.sidebar.markdown(
"""
<h1 style='text-align: center; 
            font-size: 20px; 
            color: #39ff14; 
            text-shadow: 0 0 5px #39ff14, 
                        0 0 10px #39ff14, 
                        0 0 20px #39ff14, 
                        0 0 40px #0fa, 
                        0 0 80px #0fa;'>
    Data from ShotQuality
</h1>
""",
unsafe_allow_html=True
)


