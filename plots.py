import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from defs import *
import numpy as np

x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
y = ['1', '2', '3', '4', '5', '6', '7', '8']

custom_colorscale = [
	[0, 'rgb(256, 256, 256)'],
	[0.125, 'rgb(256, 256, 256)'],

	[0.125, 'rgb(220, 20, 60)'],
	[0.25, 'rgb(220, 20, 60)'],

	[0.25, 'rgb(255, 215, 0)'],
	[0.375, 'rgb(255, 215, 0)'],

	[0.375, 'rgb(0, 100, 0)'],
	[0.5, 'rgb(0, 100, 0)'],

	[0.5, 'rgb(64, 224, 208)'],
	[0.625, 'rgb(64, 224, 208)'],

	[0.625, 'rgb(0, 0, 128)'],
	[0.75, 'rgb(0, 0, 128)'],

	[0.75, 'rgb(165, 42, 42)'],
	[0.875, 'rgb(165, 42, 42)'],

	[0.875, 'rgb(128, 0, 128)'],
	[1.0, 'rgb(128, 0, 128)']
]


def heatmap(probabilities, piece="all"):
	if piece == "all":
		z = np.argmax(probabilities, axis=0)
		data = [
		    go.Heatmap(
		        z=z,
		        colorscale=custom_colorscale
		    )
		]

		# annotations = []
		# for n, row in enumerate(z):
		# 	for m, val in enumerate(row):
		# 		var = z[n][m]
		# 		annotations.append(
		# 			dict(
		# 			text=str(val),
		# 			x=x[m], y=y[n],
		# 			xref='x1', yref='y1',
		# 			font=dict(color='white' if val > 0 else 'black'),
		# 			showarrow=False)
		# 		)

		layout = go.Layout(
		    title = "Board",
		    # annotations = annotations,
		    xaxis = dict(ticks=''),
		    yaxis = dict(ticks='', ticksuffix='  '),
		    width = 800,
		    height = 800
		)

		fig = go.Figure(data=data, layout=layout)
		plot_url = py.plot(fig, filename="heatmap-all-1")
	else:
		piece_class = piece_classes[piece]
		z = probabilities[piece_class,:,:]
		# data = [
		#     go.Heatmap(
		#         x=x,
		#         y=y,
		#         z=z
		#     )
		# ]

		# annotations = []
		# for n, row in enumerate(z):
		# 	for m, val in enumerate(row):
		# 		var = z[n][m]
		# 		annotations.append(
		# 			dict(
		# 			text=str(val),
		# 			x=x[m], y=y[n],
		# 			xref='x1', yref='y1',
		# 			font=dict(color='white' if val > 0 else 'black'),
		# 			showarrow=False)
		# 		)

		# layout = go.Layout(
		#     title = piece,
		#     annotations = probabilities[piece_class,:,:],
		#     xaxis = dict(ticks=''),
		#     yaxis = dict(ticks='', ticksuffix='  '),
		#     width = 800,
		#     height = 800
		# )

		# fig = go.Figure(data=data, layout=layout)

		z_text = np.around(z, decimals=3)
		fig = FF.create_annotated_heatmap(z, annotation_text=z_text)
		plot_url = py.plot(fig, filename="heatmap-" + piece + "-1")