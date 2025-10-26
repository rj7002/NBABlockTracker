import re
import pandas as pd
import streamlit as st
import requests
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import ast
import nba_api.stats.static.teams as nba
import unicodedata
from datetime import datetime, timedelta
import nba_api.stats.endpoints.playbyplayv3 as pbp

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
fulldf = pd.read_csv(f'{season}_blocks.csv')
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

from nba_api.stats.endpoints import leaguegamefinder

gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=nbateamid)
gamefind = gamefinder.get_data_frames()[0]

# types = pd.DataFrame(fulldf.dtypes.reset_index())
# types.columns = ['index', 'dtype']
# types['name'] = types['index'] + " - " + types["dtype"].astype(str)
# for index, row in types.iterrows():
#     st.write(row['name'])
if teamorplayer == 'Player':
    display_player_image(playerid,300,'')
    fulldf = fulldf[fulldf['secondary_name'] == player]
else:
    display_player_image2(nbateamid,300,'')
    fulldf = fulldf[fulldf['block_team_name'] == team]

nav = st.sidebar.selectbox('Navigation',['Blocks'])
if nav == 'Assists':
    st.warning('Note: Some assists may be missing due to missing tracking data')

    assist = fulldf[fulldf['second_action'] == 'assist']
    # teamids = assist['team_id'].unique()
    assist = assist[(assist['court_player_name'] == assist['shooter_name']) | (assist['court_player_name'] == assist['secondary_name'])]
    # assist[assist['team_id'] == teamid]
    teamid = int(teamid)
    assist['team_id'] = assist['team_id'].astype(int)
    assist = assist[assist['team_id'] == teamid]
    assisters = assist['secondary_name'].unique()
    # assistercounts = assist[assist['court_player_name'] == assist['secondary_name']].groupby('secondary_name')['second_action'].count().reset_index()
    # for index, row in assistercounts.iterrows():
    #     # assisters.append(f'{row['secondary_name']} ({row['second_action']})')
    #     assisters.append(f'{row['secondary_name']}')
    shooters = assist['shooter_name'].unique()
    st.sidebar.title('Filters')
    assister = st.sidebar.multiselect('Select assisters',assisters)
    shooter = st.sidebar.multiselect('Select shooters',shooters)
    period_filter = st.sidebar.selectbox(
            "Period", 
            options=["All"] + sorted(assist['period'].unique().tolist())
        )

    minutesmin, minutesmax = st.sidebar.slider(
            "Minutes", 
            min_value=int(assist['minutes'].min()), 
            max_value=int(assist['minutes'].max()), 
            value=(int(assist['minutes'].min()), int(assist['minutes'].max()))
        )
    assist = assist[
            (assist['minutes'] >= minutesmin) & 
            (assist['minutes'] <= minutesmax)
        ]
    if period_filter != "All":
            assist = assist[assist['period'] == period_filter]

    assist['play_descriptors'] = assist['play_descriptors'].apply(ast.literal_eval)

    unique_descriptors = set(descriptor for sublist in assist['play_descriptors'] for descriptor in sublist)
    unique_descriptors = sorted(unique_descriptors)
    selected_descriptors = st.sidebar.multiselect("Choose play descriptors", unique_descriptors)
    if selected_descriptors:
        assist = assist[assist['play_descriptors'].apply(lambda x: any(descriptor in x for descriptor in selected_descriptors))]
    if shooter:
        assist = assist[assist['shooter_name'].isin(shooter)]
    if assister:
        assist = assist[assist['secondary_name'].isin(assister)]

    passer = assist[(assist['court_player_name'] == assist['secondary_name'])]
    shooter = assist[(assist['court_player_name'] != assist['secondary_name'])]
    # st.write(len(passer))


    court_fig = go.Figure()
    draw_plotly_court(court_fig, show_title=False, labelticks=False, show_axis=False,
                                glayer='above', bg_color='dark gray', margins=0)

    passer['pass_distance'] = np.sqrt((passer['shot_x'] - passer['court_x'])**2 + (passer['shot_y'] - passer['court_y'])**2)
    pass_dist_min, pass_dist_max = st.sidebar.slider(
            "Pass Distance (ft)", 
            min_value=int(passer['pass_distance'].min()), 
            max_value=int(passer['pass_distance'].max()), 
            value=(int(passer['pass_distance'].min()), int(passer['pass_distance'].max()))
        )
    passer = passer[
            (passer['pass_distance'] >= pass_dist_min-1) & 
            (passer['pass_distance'] <= pass_dist_max+1)
        ]
    # st.write(len(passer))

    passer['pass_distance'] = round(passer['pass_distance'],2)
    shooter = shooter.merge(
    passer[['play_id', 'pass_distance']],
    on='play_id',
    how='left'
)
    
    shooter = shooter[
            (shooter['pass_distance'] >= pass_dist_min-1) & 
            (shooter['pass_distance'] <= pass_dist_max+1)
        ]
    shooter['court_x'] = 10*shooter['court_x']-50
    shooter['court_y'] = 10*shooter['court_y']-250
    passer['court_x'] = 10*passer['court_x']-50
    passer['court_y'] = 10*passer['court_y']-250

    passer['shot_x'] = 10*passer['shot_x']-50
    passer['shot_y'] = 10*passer['shot_y']-250
    passer['seconds'] = passer['seconds'].astype(str).str.zfill(2)
    passer['time'] = passer['minutes'].astype(str) + ':' + passer['seconds'].astype(str)
    shooter['time'] = shooter['minutes'].astype(str) + ':' + shooter['seconds'].astype(str)

    hover_template = (
        "<b>Passer<b>: %{customdata[0]}<br>" +
        "<b>Shooter<b>: %{customdata[1]}<br>" + 
        "<b>Pass Distance<b>: %{customdata[2]} ft<br>" + 
        "<b>Period<b>: %{customdata[3]}<br>" +
        "<b>Time<b>: %{customdata[4]}<br>" + 
        "<b>Shot Type<b>: %{customdata[5]} (%{customdata[6]})<br>" + 
        "<b>Game<b>: %{customdata[7]}<br>"
        "<b>Date<b>: %{customdata[8]}"
    )
    hover_template2 = (
        "<b>%{customdata[0]} to %{customdata[1]}<br>" + 
        "<b>Pass Distance<b>: %{customdata[2]}<br>" 
    )
    # st.write(len(passer))
    # st.write(len(passer['game_id_x'].unique()))
    common_play_ids = set(passer['play_id']) & set(shooter['play_id'])
    # st.write(common_play_ids)
    # passer = passer[passer['play_id'].isin(common_play_ids)].reset_index(drop=True)
    # shooter = shooter[shooter['play_id'].isin(common_play_ids)].reset_index(drop=True)
    # passer = passer.drop_duplicates(subset='play_id')
    # shooter = shooter.drop_duplicates(subset='play_id')
    arrows = st.sidebar.checkbox('Hide arrows',value=False)
    # location = st.sidebar.radio('Pass Location',['Start','End','None'])
    if arrows == False:
        for i in range(len(passer)):
            # Draw a line with an arrowhead
            court_fig.add_trace(go.Scatter(
                x=[passer['court_y'].iloc[i], passer['shot_y'].iloc[i]],  # Passer and Shooter x coordinates
                y=[passer['court_x'].iloc[i], passer['shot_x'].iloc[i]],  # Passer and Shooter y coordinates
                mode='lines+markers',
                line=dict(color='gold', width=2, shape='linear'),  # Arrow properties
                marker=dict(size=12, color='gold',symbol='arrow',angleref="previous",opacity=1),  # Optional: Add a marker at the passer's location
                opacity=0.5,
                # customdata=passer[['court_player_name','shooter_name']].iloc[i],  # Use customdata for makes only
                hoverinfo='none',  # Set hoverinfo to text
                # hovertemplate=hover_template
                # name=f'Pass to Shooter {i+1}',
            ))
    

    # if location == 'Start':
    court_fig.add_trace(go.Scatter(
        x=passer['court_y'], 
        y=passer['court_x'], 
        mode='markers',
        marker=dict(size=10, color='red', opacity=1,symbol='star'),
        name='',
        customdata=passer[['court_player_name','shooter_name','pass_distance','period','time','action_type','play_descriptors','game_name','game_datetime_utc']],  # Use customdata for makes only
        hoverinfo='text',  # Set hoverinfo to text
        hovertemplate=hover_template
    ))
        # court_fig.add_trace(go.Scatter(
        #     x=5-passer['court_y'], 
        #     y=passer['court_x'], 
        #     mode='markers',
        #     marker=dict(size=10, color='blue', opacity=1,symbol='circle'),
        #     name='',
        #     customdata=passer[['court_player_name','shooter_name','pass_distance','period','time','action_type','play_descriptors','game_name','game_datetime_utc']],  # Use customdata for makes only
        #     hoverinfo='text',  # Set hoverinfo to text
        #     hovertemplate=hover_template
        # ))
    # if location == 'End':
    #     court_fig.add_trace(go.Scatter(
    #         x=shooter['court_y'], 
    #         y=shooter['court_x'], 
    #         mode='markers',
    #         marker=dict(size=10, color='green', opacity=1,symbol='circle'),
    #         name='',
    #         customdata=shooter[['secondary_name','shooter_name','pass_distance','period','time','action_type','play_descriptors','game_name','game_datetime_utc']],  # Use customdata for makes only
    #         hoverinfo='text',  # Set hoverinfo to text
    #         hovertemplate=hover_template
    #     ))
    c1,c2= st.columns(2)
    with c1:    
        eventdata = st.plotly_chart(court_fig,use_container_width=True,on_select='rerun')
        # st.write(eventdata)
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
            nbagameid = str(nbagameid)[2:]
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
            actionid = pbp["actionNumber"].iloc[0]
            url = f'https://api.databallr.com/api/get_video/{nbagameid}/{actionid}'
            response = requests.get(url)
            if response.status_code == 200:
                video_url = response.text.strip()
                st.markdown(
                f"""
                <div style="text-align: center;">
                    <video controls autoplay width="500" style="margin: auto;">
                        <source src="{video_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                """,
                unsafe_allow_html=True
            )
            else:
                st.error('No video found')
            # st.write(pbp)

    court_fig2 = go.Figure()
    draw_plotly_court(court_fig2, show_title=False, labelticks=False, show_axis=False,
                                glayer='above', bg_color='dark gray', margins=0)
    # if location == 'Start':
    passer1 = passer[passer['court_x'] > 425]
    passer2 = passer[passer['court_x'] <= 425]
    court_fig2.add_trace(go.Histogram2dContour(
        x=passer1['court_y'],  # Passer x-coordinates
        y=passer1['court_x'],  # Passer y-coordinates
        # colorscale=[[0, 'black'], [0.1, 'black'],[0.3, 'yellow'], [1, 'red']],  # Color scale for the contour
        colorscale=[[0, 'black'],[0.2, 'yellow'], [1, 'red']],  # Color scale for the contour
        # opacity=0.6,  # Make the contour semi-transparent
        contours=dict(
            coloring='heatmap',  # Coloring the contours based on heatmap density
            showlabels=False,  # Show contour labels (optional)
            labelfont=dict(size=12)  # Font size of the contour labels (optional)
        ),
        showlegend=False,
         hovertemplate='Assists: %{z}<extra></extra>'
    ))
    court_fig2.add_trace(go.Histogram2dContour(
        x=passer2['court_y'],  # Passer x-coordinates
        y=passer2['court_x'],  # Passer y-coordinates
        # colorscale=[[0, 'black'], [0.1, 'black'],[0.3, 'yellow'], [1, 'red']],  # Color scale for the contour
        colorscale=[[0, 'black'],[0.2, 'yellow'], [1, 'red']],  # Color scale for the contour
        # opacity=0.6,  # Make the contour semi-transparent
        contours=dict(
            coloring='heatmap',  # Coloring the contours based on heatmap density
            showlabels=False,  # Show contour labels (optional)
            labelfont=dict(size=12)  # Font size of the contour labels (optional)
        ),
        showlegend=False,
         hovertemplate='Assists: %{z}<extra></extra>'
    ))
    # if location == 'End':
    #     shooter1 = shooter[shooter['court_x'] > 425]
    #     shooter2 = shooter[shooter['court_x'] <= 425]
    #     court_fig2.add_trace(go.Histogram2dContour(
    #         x=shooter1['court_y'],  # Passer x-coordinates
    #         y=shooter1['court_x'],  # Passer y-coordinates
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
    #         x=shooter2['court_y'],  # Passer x-coordinates
    #         y=shooter2['court_x'],  # Passer y-coordinates
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
    court_fig2.update_traces(showscale=False)
    with c2:
        st.plotly_chart(court_fig2,use_container_width=False)
    ca1,ca2 = st.columns(2)
    with ca1:
        histfig = px.histogram(data_frame=passer,x='pass_distance')
        histfig.update_traces(marker_line_color='black', marker_line_width=1)
        histfig.update_layout(
        title="Pass Distance Distribution",
        xaxis_title="Pass Distance (ft)",
        yaxis_title="Count"
    )
        st.plotly_chart(histfig)
    with ca2:
        barfig = px.histogram(passer,x='period',color='action_type')
        barfig.update_traces(marker_line_color='black', marker_line_width=1)
        barfig.update_layout(title='Assist Counts by Period')
        st.plotly_chart(barfig)
    cb1,cb2 = st.columns(2)
    with cb1:
        passer2 = passer
        # passer2['play_descriptors'] = passer2['play_descriptors'].apply(ast.literal_eval)
        passer_exploded = passer2.explode('play_descriptors')
        avgpasslength = passer_exploded.groupby('play_descriptors')['pass_distance'].mean().reset_index()
        fig = px.box(passer_exploded, 
             x='play_descriptors', 
             y='pass_distance', 
             title='Pass Distance Distribution by Descriptor',
             points='outliers')

        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
    with cb2:
        if assister or teamorplayer == 'Player':
            # counts = passer['shooter_name'].value_counts().reset_index()
            # counts.columns = ['shooter_name', 'count']
            # piefig = px.pie(counts, names='shooter_name', values='count', title='Shooter Distribution')
            passer = passer.reset_index(drop=True)  # ensure index is clean
            passer['play_id'] = passer.index

            df_exploded = passer.explode('play_descriptors')

            secondary_labels = df_exploded['secondary_name'].unique().tolist()
            shooter_labels = df_exploded['shooter_name'].unique().tolist()
            play_labels = df_exploded['play_descriptors'].unique().tolist()

            labels = secondary_labels + shooter_labels + play_labels

            secondary_idx = {name: i for i, name in enumerate(secondary_labels)}
            shooter_idx = {name: i + len(secondary_labels) for i, name in enumerate(shooter_labels)}
            play_idx = {name: i + len(secondary_labels) + len(shooter_labels) for i, name in enumerate(play_labels)}


            sec_to_shooter = df_exploded.groupby(['secondary_name', 'shooter_name'])['play_id'].nunique().reset_index(name='count')

            shooter_to_play = df_exploded.groupby(['shooter_name', 'play_descriptors'])['play_id'].nunique().reset_index(name='count')

            source_sec = sec_to_shooter['secondary_name'].map(secondary_idx)
            target_shooter = sec_to_shooter['shooter_name'].map(shooter_idx)
            value_sec_shooter = sec_to_shooter['count']

            source_shooter = shooter_to_play['shooter_name'].map(shooter_idx)
            target_play = shooter_to_play['play_descriptors'].map(play_idx)
            value_shooter_play = shooter_to_play['count']

            sources = pd.concat([source_sec, source_shooter])
            targets = pd.concat([target_shooter, target_play])
            values = pd.concat([value_sec_shooter, value_shooter_play])
            piefig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values
                )
            )])

            piefig.update_layout(title_text="Passer → Shooter → Play Description Sankey", font_size=12)
        else:
            counts = passer['secondary_name'].value_counts().reset_index()
            counts.columns = ['secondary_name', 'count']
            piefig = px.pie(counts, names='secondary_name', values='count', title='Pass Distribution')
        st.plotly_chart(piefig)
if nav == 'Rebounds':
    st.warning('Note: Many rebounds may be missing due to missing tracking data')

    hover_template = (
        "<b>Rebounder<b>: %{customdata[0]}<br>" +
        "<b>Shooter<b>: %{customdata[1]}<br>" + 
        "<b>Rebound Type<b>: %{customdata[2]}<br>" + 
        "<b>Period<b>: %{customdata[3]}<br>" +
        "<b>Time<b>: %{customdata[4]}<br>" + 
        "<b>Shot Type<b>: %{customdata[5]} (%{customdata[6]})<br>" + 
        "<b>Game<b>: %{customdata[7]}<br>"
        "<b>Date<b>: %{customdata[8]}"
    )
    # st.write(fulldf.groupby('second_action')['play_id'].count())
    rebound = fulldf[(fulldf['second_action'] == 'dreb') | (fulldf['second_action'] == 'oreb')]
    # st.write(rebound.groupby('second_action')['play_id'].count())
    # st.write(fulldf.drop_duplicates(subset='play_id'))
    teamids = rebound['team_id'].unique()
    rebound = rebound[(rebound['court_player_name'] == rebound['secondary_name'])]
    rebound['distance_from_shot'] = np.sqrt((rebound['shot_x'] - rebound['court_x'])**2 + (rebound['shot_y'] - rebound['court_y'])**2)
    rebound['play_descriptors'] = rebound['play_descriptors'].apply(ast.literal_eval)

    # st.write(rebound.groupby('second_action')['play_id'].count())

    rebound['team_id'] = rebound['team_id'].astype(int)
    teamid = int(teamid)
    oreb = rebound[rebound['second_action'] == 'oreb']
    dreb = rebound[rebound['second_action'] == 'dreb']

    oreb = oreb[oreb['team_id'] == teamid]
    dreb = dreb[dreb['team_id'] != teamid]
    rebound = pd.concat([oreb,dreb],axis=0,ignore_index=True)
    # st.write(rebound.groupby('second_action')['play_id'].count())

    # st.write(len(rebound))
    # st.write(len(rebound['game_id_x'].unique()))


    rebound['court_x'] = 10*rebound['court_x']-50
    rebound['court_y'] = 10*rebound['court_y']-250
    rebound['time'] = rebound['minutes'].astype(str) + ':' + rebound['seconds'].astype(str)
    st.sidebar.title('Filters')
    rebounders = rebound['secondary_name'].unique()
    shooters = rebound['shooter_name'].unique()
    rebounder = st.sidebar.multiselect('Select rebounder',rebounders)
    shooter = st.sidebar.multiselect('Select shooters',shooters)
    period_filter = st.sidebar.selectbox(
            "Period", 
            options=["All"] + sorted(rebound['period'].unique().tolist())
        )

    minutesmin, minutesmax = st.sidebar.slider(
            "Minutes", 
            min_value=int(rebound['minutes'].min()), 
            max_value=int(rebound['minutes'].max()), 
            value=(int(rebound['minutes'].min()), int(rebound['minutes'].max()))
        )
    rebound = rebound[
            (rebound['minutes'] >= minutesmin) & 
            (rebound['minutes'] <= minutesmax)
        ]
    if period_filter != "All":
            rebound = rebound[rebound['period'] == period_filter]
    unique_descriptors = set([descriptor for sublist in rebound['play_descriptors'] for descriptor in sublist])
    unique_descriptors = sorted(unique_descriptors)
    reboundtypes = rebound['second_action'].unique()
    type = st.sidebar.multiselect('Rebound Type',reboundtypes)
    if type:
        rebound = rebound[rebound['second_action'].isin(type)]
    selected_descriptors = st.sidebar.multiselect("Choose play descriptors", unique_descriptors)
    if selected_descriptors:
        rebound = rebound[rebound['play_descriptors'].apply(lambda x: any(descriptor in x for descriptor in selected_descriptors))]
    if shooter:
        rebound = rebound[rebound['shooter_name'].isin(shooter)]
    if rebounder:
        rebound = rebound[rebound['secondary_name'].isin(rebounder)]
    
    # st.write(len(rebound))
    court_fig = go.Figure()
    draw_plotly_court(court_fig, show_title=False, labelticks=False, show_axis=False,
                                glayer='above', bg_color='dark gray', margins=0)
    court_fig.add_trace(go.Scatter(
            x=rebound['court_y'], 
            y=rebound['court_x'], 
            mode='markers',
            marker=dict(size=10, color='red', opacity=1,symbol='star-square'),
            name='',
            customdata=rebound[['court_player_name','shooter_name','second_action','period','time','action_type','play_descriptors','game_name','game_datetime_utc']],  # Use customdata for makes only
            hoverinfo='text',  # Set hoverinfo to text
            hovertemplate=hover_template
        ))
    c1,c2 = st.columns(2)
    with c1:
        # st.plotly_chart(court_fig,use_container_width=False)
        eventdata = st.plotly_chart(court_fig,use_container_width=True,on_select='rerun')
        # st.write(eventdata)
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
            nbagameid = str(nbagameid)[2:]
            # st.write(gamefind)
            # st.write(x)
            # st.write(y)
            # st.write(nbagameid)
            pbp = pbp.PlayByPlayV3(game_id=oldgameid).get_data_frames()[0]
            def fixTime(iso_str):
                # Match minutes and seconds using regex
                match = re.match(r"PT(\d+)M(\d+(?:\.\d+)?)S", iso_str)
                if not match:
                    return None  # or raise an error
                
                minutes, seconds = match.groups()

                # If seconds ends in .0, strip it
                seconds = str(float(seconds)).rstrip('0').rstrip('.') 
                
                return f"{int(minutes)}:{seconds}"
            pbp['time'] = pbp['clock'].apply(fixTime)
            # st.write(pbp)
            def parse_time_str(time_str):
                """Convert a string like '1:19.0' into a timedelta object."""
                try:
                    minutes, seconds = time_str.split(':')
                    return timedelta(minutes=int(minutes), seconds=float(seconds))
                except Exception as e:
                    print("Error parsing time:", e)
                    return None

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
            actionid = pbp["actionNumber"].iloc[0]
            url = f'https://api.databallr.com/api/get_video/{nbagameid}/{actionid}'
            response = requests.get(url)
            if response.status_code == 200:
                video_url = response.text.strip()
                st.markdown(
                f"""
                <div style="text-align: center;">
                    <video controls autoplay width="500" style="margin: auto;">
                        <source src="{video_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                """,
                unsafe_allow_html=True
            )
            else:
                st.error('No video found')
            # st.write(pbp)
    rebound1 = rebound[rebound['court_x'] > 425]
    rebound2 = rebound[rebound['court_x'] <= 425]
    court_fig2 = go.Figure()
    draw_plotly_court(court_fig2, show_title=False, labelticks=False, show_axis=False,
                                glayer='above', bg_color='dark gray', margins=0)
    court_fig2.add_trace(go.Histogram2dContour(
        x=rebound1['court_y'],  # Passer x-coordinates
        y=rebound1['court_x'],  # Passer y-coordinates
        # colorscale=[[0, 'black'], [0.1, 'black'],[0.3, 'yellow'], [1, 'red']],  # Color scale for the contour
        colorscale=[[0, 'black'],[0.2, 'yellow'], [1, 'red']],  # Color scale for the contour
        # opacity=0.6,  # Make the contour semi-transparent
        contours=dict(
            coloring='heatmap',  # Coloring the contours based on heatmap density
            showlabels=False,  # Show contour labels (optional)
            labelfont=dict(size=12)  # Font size of the contour labels (optional)
        ),
        showlegend=False,
         hovertemplate='Rebounds: %{z}<extra></extra>'
    ))
    court_fig2.add_trace(go.Histogram2dContour(
        x=rebound2['court_y'],  # Passer x-coordinates
        y=rebound2['court_x'],  # Passer y-coordinates
        # colorscale=[[0, 'black'], [0.1, 'black'],[0.3, 'yellow'], [1, 'red']],  # Color scale for the contour
        colorscale=[[0, 'black'],[0.2, 'yellow'], [1, 'red']],  # Color scale for the contour
        # opacity=0.6,  # Make the contour semi-transparent
        contours=dict(
            coloring='heatmap',  # Coloring the contours based on heatmap density
            showlabels=False,  # Show contour labels (optional)
            labelfont=dict(size=12)  # Font size of the contour labels (optional)
        ),
        showlegend=False,
        hovertemplate='Rebounds: %{z}<extra></extra>'
    ))
    court_fig2.update_traces(showscale=False)
    with c2:
        st.plotly_chart(court_fig2,use_container_width=False)
    ca1,ca2 = st.columns(2)
    with ca1:
        histfig = px.histogram(data_frame=rebound,x='second_action',color='period')
        histfig.update_traces(marker_line_color='black', marker_line_width=1)
        histfig.update_layout(
        title="Rebound Counts"
    )
        st.plotly_chart(histfig)
    with ca2:
        barfig = px.histogram(rebound,x='period',color='action_type')
        barfig.update_traces(marker_line_color='black', marker_line_width=1)
        barfig.update_layout(title='Rebound Counts by Period')
        st.plotly_chart(barfig)
    cb1,cb2 = st.columns(2)
    with cb1:
        rebound2 = rebound
        # passer2['play_descriptors'] = passer2['play_descriptors'].apply(ast.literal_eval)
        rebound_exploded = rebound2.explode('play_descriptors')
        fig = px.box(rebound_exploded, 
             x='play_descriptors', 
             y='distance_from_shot', 
             title='Distance From Shot Distribution by Descriptor',
             points='outliers')

        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
    with cb2:
        if rebounder or teamorplayer == 'Player':
            counts = rebound['shooter_name'].value_counts().reset_index().head(10)
            counts.columns = ['shooter_name', 'count']
            piefig = px.pie(counts, names='shooter_name', values='count', title='Top 10 Shooter Rebound Distribution')
        else:
            counts = rebound['secondary_name'].value_counts().reset_index()
            counts.columns = ['secondary_name', 'count']
            piefig = px.pie(counts, names='secondary_name', values='count', title='Rebound Distribution by Player')
        st.plotly_chart(piefig)
if nav == 'Blocks':
    st.warning('Note: Some blocks may be missing due to missing tracking data')

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
    court_fig.add_trace(go.Scatter(
            x=block['shot_y'], 
            y=block['shot_x'], 
            mode='markers',
            marker=dict(size=10, color='red', opacity=1,symbol='x'),
            name='',
            customdata=block[['court_player_name','shooter_name','distance_covered','period','time','action_type','play_descriptors','game_name','game_datetime_utc','ClosestDefVelBlockAng']],  # Use customdata for makes only
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
    # st.write(eventdata)
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
        nbagameid = str(nbagameid)[2:]
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
        actionid = pbp["actionNumber"].iloc[0]
        url = f'https://api.databallr.com/api/get_video/{nbagameid}/{actionid}'
        response = requests.get(url)
        if response.status_code == 200:
            video_url = response.text.strip()
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <video controls autoplay width="500" style="margin: auto;">
                        <source src="{video_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error('No video found')
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
# import requests
# import pandas as pd
# import streamlit as st
# # Define the API URL
# url = "https://api.databallr.com/api/supabase/sqpbp?game_id=22401064&limit=100000"

# # Send a GET request to fetch the raw JSON data
# response = requests.get(url)
# data = response.json()

# # Convert the JSON data into a Pandas DataFrame
# df = pd.json_normalize(data)

# # Display the first few rows of the DataFrame
# st.write(df)


