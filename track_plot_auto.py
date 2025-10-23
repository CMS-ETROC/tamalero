import os
import struct
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
pio.renderers.default = 'browser'
from tamalero.FIFO import merge_words
from tamalero.DataFrame import DataFrame

# --- Configuration ---
INPUT_DATA_PATH = '/home/roy/yf_temp/tamalero/cosmic_run/10-17_14-16-59'
DEFAULT_Z_SPACING = 20.0  # mm
BCID_WINDOW = 2
AUTOPLAY_INTERVAL_MS = 3000 



def get_chip_id_from_elink(elink):
    elink_map = {0: 0, 4: 1, 8: 2, 12: 3}
    return elink_map.get(elink, -1)

def get_3d_coordinates(chip_id, row, col, z_spacing_mm):
    return (float(col), float(row), float(chip_id * z_spacing_mm))

def parse_single_file(dat_file_path):
    print(f"Parsing data file from : {os.path.basename(dat_file_path)}")
    try:
        with open(dat_file_path, 'rb') as f:
            raw_content = f.read()
        num_words, raw_words = len(raw_content) // 4, struct.unpack(f'<{len(raw_content) // 4}I', raw_content)
    except Exception as e:
        print(f"Error: unable to read from {dat_file_path}: {e}"); 
        return []
    
    merged64_words = merge_words(list(raw_words))
    df_parser = DataFrame(version='ETROC2')
    file_hits, current_bcid = [], -1
    for word in merged64_words:
        data_type, parsed_data = df_parser.read(word, quiet=True)
        if data_type == 'header': current_bcid = parsed_data.get('bcid', current_bcid)
        elif data_type == 'data':
            elink, row, col, toa, tot, cal = (parsed_data.get(k) for k in ['elink', 'row_id', 'col_id', 'toa', 'tot', 'cal'])
            if all(v is not None for v in [elink, row, col, toa]):
                chip_id = get_chip_id_from_elink(elink)
                if chip_id != -1:
                    file_hits.append({
                        "chip_id": chip_id, "row": row, "col": col, 
                        "bcid": current_bcid, "toa": toa,
                        "tot": tot if tot is not None else 0,
                        "cal": cal if cal is not None else 0
                    })
    print(f"Successfully parsed {len(file_hits)} hits in total")
    return file_hits

def cluster_and_find_tracks(all_hits, z_spacing_mm):
    if not all_hits: return []
    all_hits.sort(key=lambda h: (h['bcid'], h['toa']))
    clustered_events, current_event = [], []
    for hit in all_hits:
        if not current_event or (hit['bcid'] - current_event[-1]['bcid'] <= BCID_WINDOW):
            current_event.append(hit)
        else:
            clustered_events.append(current_event); current_event = [hit]
    if current_event: clustered_events.append(current_event)
    track_candidates = []
    for event in clustered_events:
        unique_chip_ids = set(h['chip_id'] for h in event)
        if len(unique_chip_ids) >= 3 and len(unique_chip_ids) == len(event):
            for hit in event: hit['pos'] = get_3d_coordinates(hit['chip_id'], hit['row'], hit['col'], z_spacing_mm)
            track_candidates.append(event)
    return track_candidates

def load_and_process_path(path, z_spacing):

    if not os.path.exists(path):
        print(f"Error: Path does not exist '{path}'")
        return []
    
    dat_files_to_process = []
    if os.path.isdir(path):
        print(f"Loading data from folder: {os.path.basename(path)}")
        dat_files = sorted([f for f in os.listdir(path) if f.endswith('.dat')])
        if not dat_files:
            print(f"Warning: No .dat files found in folder '{path}'.")
            return []
        print(f"Found {len(dat_files)} .dat files.")
        for filename in dat_files:
            dat_files_to_process.append(os.path.join(path, filename))
            
    elif os.path.isfile(path):
        print(f"=== Loading data from single file: {os.path.basename(path)} ===")
        dat_files_to_process.append(path)
    
    else:
        print(f"Error: Path '{path}' is neither a file nor a folder.")
        return []
    # Store all tracks found across all files
    master_track_list = []
    total_hits_parsed = 0

    for file_path in dat_files_to_process:
        # Parse single file
        hits_from_file = parse_single_file(file_path)
        total_hits_parsed += len(hits_from_file)
        
        # Cluster and find tracks from this file's hits
        print("--- Clustering and finding track candidates ---")
        tracks_from_file = cluster_and_find_tracks(hits_from_file, z_spacing)
        print(f"Found {len(tracks_from_file)} high-quality tracks in {os.path.basename(file_path)}.")
        
        # Add found tracks to master list
        if tracks_from_file:
            master_track_list.extend(tracks_from_file)
        print("-" * 50)
    print(f"\nAll files processed. Total hits parsed: {total_hits_parsed}")
    print(f"Total high-quality tracks found: {len(master_track_list)}")
    return master_track_list

ALL_TRACKS = load_and_process_path(INPUT_DATA_PATH, DEFAULT_Z_SPACING)
TRACKS_3_HIT = [t for t in ALL_TRACKS if len(t) == 3]
TRACKS_4_HIT = [t for t in ALL_TRACKS if len(t) == 4]

def fit_3d_line(points):
    if points.shape[0] < 2: raise ValueError("Need at least 2 points to fit a line")
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    return centroid, vh[0]

def create_hybrid_figure(track, z_spacing):
    fig_data = []
    board_labels = ["ET2.02-PT-IH13", "ET2.02-PT-IH11", "ET2.02-PT-IH7", "ET2.02-PT-IH12"]
    for i, label in enumerate(board_labels):
        fig_data.append(go.Scatter3d(
            x=[-1.5], y=[8], z=[i * z_spacing],
            text=[label], mode='text', textfont=dict(size=12, color='white')
        ))

    grid_line_color = '#888888'
    pixel_coords = np.arange(-0.5, 16.5, 1)
    xx, yy = np.meshgrid(pixel_coords, pixel_coords)
    for i in range(4):
        z = i * z_spacing
        zz = np.full_like(xx, z)
        fig_data.append(go.Surface(x=xx, y=yy, z=zz, colorscale=[[0, '#00FFFF'], [1, '#00FFFF']], 
                                   opacity=0.05, showscale=False))
        for k in np.arange(-0.5, 16.5, 1):
            fig_data.append(go.Scatter3d(x=[-0.5, 15.5], y=[k, k], z=[z, z], mode='lines', 
                                         line=dict(color=grid_line_color, width=1)))
            fig_data.append(go.Scatter3d(x=[k, k], y=[-0.5, 15.5], z=[z, z], mode='lines', 
                                         line=dict(color=grid_line_color, width=1)))
            
    for hit in track:
        x, y, z = hit['pos']
        hover_text = (
            f"<b>Chip {hit['chip_id']}</b><br>"
            f"Row: {hit['row']}, Col: {hit['col']}<br>"
            f"<br><b>TDC Data:</b><br>"
            f"ToA (raw): {hit['toa']}<br>"
            f"ToT: {hit['tot']}<br>"
            f"Cal: {hit['cal']}<br>"
        )
        fig_data.append(go.Mesh3d(
            x=[x-0.5, x+0.5, x+0.5, x-0.5], 
            y=[y-0.5, y-0.5, y+0.5, y+0.5], 
            z=[z, z, z, z],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color='#FF1493', opacity=0.9,
            text=hover_text, hoverinfo='text',
            name=f'Hit @ Chip {hit["chip_id"]}'
        ))

    try:
        points = np.array([h['pos'] for h in track])
        centroid, direction = fit_3d_line(points)
        z_min_ext, z_max_ext = -1.5 * z_spacing, 4.5 * z_spacing
        if abs(direction[2]) > 1e-6:
            t_min = (z_min_ext - centroid[2]) / direction[2]
            t_max = (z_max_ext - centroid[2]) / direction[2]
            p_min, p_max = centroid + t_min * direction, centroid + t_max * direction
            fig_data.append(go.Scatter3d(x=[p_min[0], p_max[0]], y=[p_min[1], p_max[1]], z=[p_min[2], p_max[2]],
                                         mode='lines', line=dict(color='#00FF00', width=6)))
    except Exception as e:
        print(f"Could not fit line for track: {e}")
    
    z_range = 3 * z_spacing
    xy_range = 16
    z_aspect = z_range / xy_range
    
    layout = go.Layout(
        title=dict(text=f'Reconstructed Particle Track ({len(track)} hits, Z-spacing={z_spacing}mm)', x=0.5),
        scene=dict(
            xaxis=dict(title='Column (X)', showbackground=False, gridcolor='#444', zeroline=False, range=[-2, 17]),
            yaxis=dict(title='Row (Y)', showbackground=False, gridcolor='#444', zeroline=False, range=[-2, 17]),
            zaxis=dict(title=f'Layer Depth (Z, mm)', showbackground=False, gridcolor='#444', zeroline=False),
            aspectratio=dict(x=1, y=1, z=z_aspect),
            camera_eye=dict(x=1.6, y=-1.6, z=1.8)
        ),
        template='plotly_dark',
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return go.Figure(data=fig_data, layout=layout)

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#FFFFFF', 'fontFamily': 'Arial'}, children=[
    html.H1('Cosmic Ray Track Explorer', style={'textAlign': 'center', 'padding': '10px'}),
    html.Div([
        html.Span(f'Z-Spacing: {DEFAULT_Z_SPACING} mm', 
                  style={'fontSize': '1.1em', 'color': '#00FF00', 'marginRight': '20px'}),
        html.Span('💡 Hover over hits to see TDC data (ToA/ToT/Cal)', 
                  style={'fontSize': '0.95em', 'color': '#FFD700'})
    ], style={'textAlign': 'center', 'padding': '5px'}),
    dcc.Store(id='track-state', data={'index': 0, 'filter': 'all'}),
    
    dcc.Interval(
        id='interval-component',
        interval=AUTOPLAY_INTERVAL_MS,
        n_intervals=0,
        disabled=True, 
    ),

    html.Div([
        html.Div([
            html.Label('Filter Tracks:', style={'paddingRight':'10px'}),
            dcc.RadioItems(id='filter-radio',
                options=[{'label': f'All ({len(ALL_TRACKS)})', 'value': 'all'},
                         {'label': f'3-Hit ({len(TRACKS_3_HIT)})', 'value': '3'},
                         {'label': f'4-Hit ({len(TRACKS_4_HIT)})', 'value': '4'}],
                value='all', labelStyle={'display': 'inline-block', 'marginRight': '20px'})
        ], style={'flex': '1', 'textAlign': 'center'}),
        html.Div([
            html.Button('Previous Track', id='prev-button', n_clicks=0),
            html.Span(id='track-counter', style={'margin': '0 20px', 'fontSize': '1.2em', 'verticalAlign': 'middle'}),
            html.Button('Next Track', id='next-button', n_clicks=0),
            html.Button('▶️ Autoplay', id='play-pause-button', n_clicks=0, style={'marginLeft': '25px'}),
        ], style={'flex': '1', 'textAlign': 'center'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '10px', 'borderBottom': '1px solid #444'}),
    dcc.Graph(id='track-graph', style={'height': '75vh'}),
    html.Div(id='hit-info-panel', style={
        'padding': '15px', 'backgroundColor': '#2A2A2A',
        'borderTop': '2px solid #444', 'minHeight': '80px'
    }, children=[
        html.H4('Hit Details', style={'marginTop': '0', 'color': '#00FF00'}),
        html.Div(id='hit-details', children='Hover over a hit to see detailed TDC information')
    ])
])

@app.callback(
    [Output('interval-component', 'disabled'),
     Output('play-pause-button', 'children')],
    [Input('play-pause-button', 'n_clicks')],
    [State('interval-component', 'disabled')]
)
def toggle_autoplay(n_clicks, is_disabled):
    if n_clicks == 0:
        return True, '▶️ Autoplay'
    return not is_disabled, '▶️ Autoplay' if not is_disabled else '⏸️ Pause'

@app.callback(
    [Output('track-graph', 'figure'),
     Output('track-counter', 'children'),
     Output('track-state', 'data')],
    [Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks'),
     Input('filter-radio', 'value'),
     Input('interval-component', 'n_intervals')],
    [State('track-state', 'data')]
)
def update_track_view(prev_clicks, next_clicks, filter_value, n_intervals, current_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger_id = 'filter-radio'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if filter_value == '3': active_tracks = TRACKS_3_HIT
    elif filter_value == '4': active_tracks = TRACKS_4_HIT
    else: active_tracks = ALL_TRACKS

    if trigger_id == 'filter-radio': index = 0
    else:
        index = current_state['index']
        if trigger_id in ['next-button', 'interval-component']: index += 1
        elif trigger_id == 'prev-button': index -= 1
    
    if not active_tracks: 
        return go.Figure().update_layout(template='plotly_dark', title='No tracks match filter'), "No tracks match filter", {'index': 0, 'filter': filter_value}
    
    index = index % len(active_tracks)
    
    figure = create_hybrid_figure(active_tracks[index], DEFAULT_Z_SPACING)
    counter_text = f"Track {index + 1} of {len(active_tracks)}"
    new_state = {'index': index, 'filter': filter_value}
    
    return figure, counter_text, new_state

@app.callback(
    Output('hit-details', 'children'),
    [Input('track-graph', 'hoverData')],
    [State('track-state', 'data')]
)
def display_hover_data(hoverData, current_state):
    if not hoverData:
        return 'Hover over a hit to see detailed TDC information'
    
    try:
        filter_value = current_state.get('filter', 'all')
        if filter_value == '3': active_tracks = TRACKS_3_HIT
        elif filter_value == '4': active_tracks = TRACKS_4_HIT
        else: active_tracks = ALL_TRACKS
        
        index = current_state.get('index', 0)
        if not active_tracks: return 'No track data available'
        
        point = hoverData['points'][0]
        hover_text = point.get('text', '')
        
        if hover_text:
            return html.Div([
                html.P(line, style={'margin': '5px 0'}) 
                for line in hover_text.replace('<br>', '\n').replace('<b>', '').replace('</b>', '').split('\n')
                if line.strip()
            ])
        return 'Hover over a hit pixel to see details'
    except Exception as e:
        return f'Error displaying data: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)