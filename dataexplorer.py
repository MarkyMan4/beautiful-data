import streamlit as st
import pandas as pd
import os


def left_side():
	data = None

	st.sidebar.markdown('### Select a data set')
	up_file = st.sidebar.file_uploader('Upload a file', type='csv')

	if up_file:
		data = pd.read_csv(up_file)

	if st.sidebar.checkbox('Use sample data'):
		data = pd.read_csv('data/iris_data.csv')

	return data

def display_data(data):
	if st.checkbox('Show raw data'):
		st.write(data)

	if st.checkbox('Line chart'):
		st.line_chart(data)

def main():
	st.markdown('# Data Explorer :tada:')

	data = left_side()
	if data is not None:
		display_data(data)
	else:
		st.subheader('No file chosen')

if __name__ == '__main__':
	main()