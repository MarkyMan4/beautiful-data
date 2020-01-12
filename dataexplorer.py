import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from iso_regression import IsoRegressor
from lin_regression import LinRegressor
from poly_regression import PolyRegressor
import os


def left_side():
	data = None
	selected_cols = []

	st.sidebar.markdown('### Select a data set')
	up_file = st.sidebar.file_uploader('Upload a file', type='csv')

	if up_file:
		data = pd.read_csv(up_file)

	if st.sidebar.checkbox('Use sample data'):
		data = pd.read_csv('data/iris_data.csv')

	c_boxes = []

	if data is not None:
		for i in range(len(data.columns)):
			c_boxes.append(st.sidebar.checkbox(data.columns[i]))

		for i in range(len(c_boxes)):
			if c_boxes[i]:
				selected_cols.append(data.columns[i])

	return data, selected_cols

def scatter(data, selected_cols):
	dim = len(selected_cols) * 2
	fig = plt.figure(figsize=(dim, dim))
	ax = fig.gca()
	scatter_matrix(data, ax=ax)
	st.write(fig)

def regression_options(data):
	reg_select = st.selectbox('Select a type of regression', ['Linear', 'Isotonic', 'Polynomial'])
	model = None

	x = st.radio('X', data.columns)
	y = st.radio('Y', data.columns)

	if reg_select == 'Linear':
		model = LinRegressor(data)
	elif reg_select == 'Isotonic':
		model = IsoRegressor(data)
	elif reg_select == 'Polynomial':
		model = PolyRegressor(data)

	fig = model.get_graph(x, y)
	st.write(fig)

	input = st.text_input(f'Enter a value for {x} to predict {y}')

	if st.button('Predict'):
		pred = model.make_prediction(float(input))

		# need to get the first value since the prediction is returned as a list
		if reg_select == 'Polynomial':
			pred = pred[0][0]
		else:
			pred = pred[0]
		st.write(f'Predicted {y} = {round(pred, 3)}')

def show_stats(data):
	dataset = st.radio('Data', data.columns)

	set_for_stats = data[[dataset]]

	mean = set_for_stats.mean()
	mean = mean[dataset]

	median = set_for_stats.median()
	median = median[dataset]

	std_dev = set_for_stats.std()
	std_dev = std_dev[dataset]

	st.write(f'Mean: {round(mean, 3)}')
	st.write(f'Median: {round(median, 3)}')
	st.write(f'Standard Deviation: {round(std_dev, 3)}')

def show_map(data):
	dim = st.selectbox('View option', ['2d', '3d'])

	if dim == '2d':
		st.map(data)
	elif dim == '3d':
		midpoint = (np.average(data["lat"]), np.average(data["lon"]))

		st.deck_gl_chart(
		    viewport={
		        "latitude": midpoint[0],
		        "longitude": midpoint[1],
		        "zoom": 11,
		        "pitch": 50,
		    },
		    layers=[
		        {
		            "type": "HexagonLayer",
		            "data": data,
		            "radius": 100,
		            "elevationScale": 7,
		            "elevationRange": [0, 2000],
		            "pickable": True,
		            "extruded": True,
		        }
		    ],
		)

def show_data(data, selected_cols):
	if len(selected_cols) <= 0:
		st.subheader('Select at least one column')

	else:
		display_data = data[selected_cols]

		if st.checkbox('Show raw data'):
			st.write(display_data)

		if st.checkbox('Line chart'):
			st.line_chart(display_data)

		if len(selected_cols) >= 2 and st.checkbox('Variable relationship (scatter matrix)'):
			scatter(display_data, selected_cols)

		if st.checkbox('Regression'):
			regression_options(display_data)

		if st.checkbox('Stats'):
			show_stats(display_data)

		if ('lat' in selected_cols) and ('lon' in selected_cols):
			if st.checkbox('Show on map'):
				show_map(display_data)


def main():
	st.markdown('# Data Explorer :chart_with_upwards_trend::mag:')

	data, selected_cols = left_side()
	if data is not None:
		show_data(data, selected_cols)
	else:
		st.subheader('No file chosen')

if __name__ == '__main__':
	main()