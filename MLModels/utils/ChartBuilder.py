from plotly.graph_objs import Layout, Scatter, Figure, Marker
import matplotlib.pyplot as plt
import random

import plotly
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.tools as tls

import numpy as np
import bokeh.plotting as bk_plotting
import bokeh.models as bk_models
from bokeh.layouts import gridplot
from bokeh.layouts import row
bk_plotting.output_notebook()

import cufflinks as cf
# import plotly.graph_objs.layout.scene.Annotation
cf.go_offline()


def plot_eeg_bokeh(raw, excluded_channels, resolution_hz, scaler=None):

    # adapted from https://stackoverflow.com/a/28073161/2466781
    n_channels = len(all_channels) - len(excluded_channels)
    start, stop = raw.time_as_index([0, 500])  # manually set stop to 500s
    picks = range(n_channels)
    data, times = raw[picks[:n_channels], start:stop]
    
    print(np.asarray(data).shape)
    if scaler: 
        data = standardize_dataset(data, scaler['means'], scaler['stdevs']) 

    subplots = []
    for ii, channel in enumerate(raw.info['ch_names']):
        if channel not in excluded_channels:
            
            y_data = data.T[:, ii]
                
            source = bk_models.ColumnDataSource(data=dict(x=times, y=y_data))
            title = bk_models.annotations.Title(align='center')
            title.text = channel
            # x_range = bk_models.Range1d(start=start, end=stop)
            y_range = bk_models.Range1d(start=min(y_data), end=max(y_data))

            p1 = bk_models.Plot(  # x_range=x_range,
                                y_range=y_range,
                                min_border=2,
                                plot_width=250,
                                plot_height=250,
                                toolbar_location="below")
            p1.add_glyph(source,
                         bk_models.glyphs.Line(x='x',
                                               y='y',
                                               line_color='black',
                                               line_width=2))
            p1.add_layout(title, "left")

            # add the axes
            x_axis = bk_models.LinearAxis()
            p1.add_layout(x_axis, 'below')
            y_axis = bk_models.LinearAxis()
            p1.add_layout(y_axis, 'left')
            if ii == len(raw.info['ch_names']) - 1:
                p1.yaxis.visible = False
            else:
                p1.axis.visible = False

            # add the grid
            p1.add_layout(bk_models.Grid(dimension=1, ticker=x_axis.ticker))
            p1.add_layout(bk_models.Grid(dimension=0, ticker=y_axis.ticker))
            subplots.append(p1)

            # add the tools
            # p1.add_tools(bk_models.PreviewSaveTool())

    grid = gridplot(subplots, ncols=1, plot_width=750, plot_height=40, toolbar_location="below")
    # bk_plotting.show(row(p1, p2))
    
    bk_plotting.show(grid)


# adapted from https://stackoverflow.com/a/28073161/2466781

def plot_processed_eeg_bokeh(raw, processed_data, excluded_channels, all_channels):
    n_channels = len(all_channels) - len(excluded_channels)
    start, stop = raw.time_as_index([0, 500])  # manually set stop to 500s
    picks = range(n_channels)
    data, times = raw[picks[:n_channels], start:stop]
    data = processed_data


    subplots = []
    for ii, channel in enumerate(raw.info['ch_names']):
        if channel not in excluded_channels:

            y_data = data.T[ii]

            source = bk_models.ColumnDataSource(data=dict(x=times,
                                                          y=y_data))

            title = bk_models.annotations.Title(align='center')
            title.text = channel
            x_range = bk_models.Range1d(start=start, end=stop)
            y_range = bk_models.Range1d(start=min(y_data), end=max(y_data))

            p1 = bk_models.Plot(  # x_range=x_range,
                y_range=y_range,
                min_border=2,
                plot_width=250,
                plot_height=250,
                toolbar_location="below")
            p1.add_glyph(source,
                         bk_models.glyphs.Line(x='x',
                                               y='y',
                                               line_color='black',
                                               line_width=2))
            p1.add_layout(title, "left")

            # add the axes
            x_axis = bk_models.LinearAxis()
            p1.add_layout(x_axis, 'below')
            y_axis = bk_models.LinearAxis()
            p1.add_layout(y_axis, 'left')
            if ii == len(raw.info['ch_names']) - 1:
                p1.yaxis.visible = False
            else:
                p1.axis.visible = False

            # add the grid
            p1.add_layout(bk_models.Grid(dimension=1, ticker=x_axis.ticker))
            p1.add_layout(bk_models.Grid(dimension=0, ticker=y_axis.ticker))
            subplots.append(p1)

            # add the tools
            # p1.add_tools(bk_models.PreviewSaveTool())

    grid = gridplot(subplots, ncols=1, plot_width=750, plot_height=40, toolbar_location="below")
    # bk_plotting.show(row(p1, p2))

    bk_plotting.show(grid)


def plot_examples(file_name, excluded_channels, resolution_hz):
    patient_id = file_name.split("\\")[-1][:-4]
    print('Raw Data')
    raw = get_raw_eeg_mne(file_name, resolution_hz)
    # events = mne.find_events(raw, stim_channel=raw.ch_names, initial_event=True, consecutive=True)
    raw.plot()
    df = raw.to_data_frame()
    print('Shape: ', df.shape)

    print('Cleaned Data ')
    print('Excluding channels {}; '.format(", ".join(excluded_channels)))
    print('''Removing first 120s of data; and last 120s of shortest sample. 
             Limiting all other samples to range of shortest sample.''')
    cleaned = get_raw_eeg_mne(file_name, resolution_hz, tmin=120, tmax=minimum_duration - 120,
                              exclude=excluded_channels)
    cleaned.crop()
    cleaned.plot()
    print('Shape: ', cleaned.to_data_frame().shape)


# cl_layout = go.Layout(width=950, height=800)

# adapted from https://plot.ly/python/v3/ipython-notebooks/mne-tutorial/
def plot_eeg_plotly(raw, excluded_channels, resolution_hz):
    n_channels = len(all_channels) - len(excluded_channels)
    picks = range(n_channels)
    start, stop = raw.time_as_index([0, -1])

    data, times = raw[picks[:n_channels], start:stop]
    ch_names = [raw.info['ch_names'][p] for p in picks[:n_channels]]

    step = 1. / n_channels
    kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)
    mc = 'rgb(27,61,120)'
    # create objects for layout and traces
    layout = Layout(yaxis=go.layout.YAxis(kwargs), showlegend=False)
    layout.update({'yaxis%d' % (0 + 1): go.layout.YAxis(kwargs), 'showlegend': False})
    traces = [Scatter(x=times, y=data.T[:, 0], marker_color=mc)]

    # loop over the channels
    for ii in range(1, n_channels):
        kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
        layout.update({'yaxis%d' % (ii + 1): go.layout.YAxis(kwargs), 'showlegend': False})
        traces.append(Scatter(x=times, y=data.T[:, ii], marker_color=mc, yaxis='y%d' % (ii + 1)))

    # add channel names using Annotations
    annotations = [go.layout.Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
                                        text=ch_name, showarrow=False)
                   for ii, ch_name in enumerate(ch_names)]
    layout.update(annotations=annotations)
    traces.reverse()  # set the fist trace to the bottom of the plot since it is the only one with x_axis

    # set the size of the figure and plot it
    layout.update(autosize=False, width=900, height=400)
    fig = Figure(data=traces, layout=layout)
    fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='black', ticklen=10, side='top')

    iplot(fig, filename='shared x_axis')


def plot_examples_lib(file_name, excluded_channels, resolution_hz, scaler=None):
    print('\nCleaned Data')
    print('Excluding channels {}; '.format(", ".join(excluded_channels)))
    print('''Removing first 120s of data; and last 120s of shortest sample.
           Limiting all other samples to range of shortest sample.''')

    cleaned = get_raw_eeg_mne(file_name, resolution_hz, tmin=120, tmax=minimum_duration - 120,
                              exclude=excluded_channels)
    cleaned.crop()
    plot_eeg_bokeh(cleaned, excluded_channels, resolution_hz, scaler=scaler)


def plot_processed_examples_lib(cleaned, processed_data, excluded_channels):
    print('\nCleaned and Processed Data')
    print('Excluding channels {}; '.format(", ".join(excluded_channels)))
    print('''Removing first 120s of data; and last 120s of shortest sample.
              Limiting all other samples to range of shortest sample.''')
    plot_processed_eeg_bokeh(cleaned, processed_data, excluded_channels)


# Plot the EEG of a random patient and a random control
def plot_random_participants(data, ignore_list, excluded_channels, all_channels, random_selection):

    print("Example of Input Data From Random Control")
    rand_control_id = random_selection['hc_id']

    print('Control subject ID h{:02d} '.format(rand_control_id))
    mne_raw_hc = random_selection['hc_raw_eeg']
    mne_raw_hc.crop()  # cleaned and cropped data
    plot_processed_eeg_bokeh(mne_raw_hc, data['hc_data'][rand_control_id -1][:-1],
                             excluded_channels, all_channels)

    print('Example of Input Data From Random Patient')
    rand_patient_id = random_selection['sz_id']
    print('Sz patient ID s{:02d}'.format(rand_patient_id))
    mne_raw_sz = random_selection['sz_raw_eeg']
    mne_raw_sz.crop()
    plot_processed_eeg_bokeh(mne_raw_sz, data['sz_data'][rand_patient_id -1][:-1],
                             excluded_channels, all_channels)

    print('Ignored files: ')
    print(",".join(ignore_list))
