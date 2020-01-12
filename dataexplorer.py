import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
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

def scatter_it(data, selected_cols):
	dim = len(selected_cols) * 2
	fig = plt.figure(figsize=(dim, dim))
	ax = fig.gca()
	scatter_matrix(data, ax=ax)
	st.write(fig)

def display_data(data, selected_cols):
	if len(selected_cols) <= 0:
		st.subheader('Select at least one column')

	else:
		display_data = data[selected_cols]

		if st.checkbox('Show raw data'):
			st.write(display_data)

		if st.checkbox('Line chart'):
			st.line_chart(display_data)

		if len(selected_cols) >= 2 and st.checkbox('Variable relationship (scatter matrix)'):
			scatter_it(display_data, selected_cols)

def main():
	st.markdown('# Data Explorer :chart_with_upwards_trend::smiley:')

	data, selected_cols = left_side()
	if data is not None:
		display_data(data, selected_cols)
	else:
		st.subheader('No file chosen')

if __name__ == '__main__':
	main()