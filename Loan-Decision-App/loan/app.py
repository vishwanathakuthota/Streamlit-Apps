import streamlit as st
import numpy as np 
import pandas as pd
import warnings
#import plotly.express as px
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import *
from scipy.stats import norm, skew #for some statistics
import pandas_bokeh
pandas_bokeh.output_notebook()
import base64

# import ml library
#from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier


#Others
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, KFold

# list number of files
#import os
#os.chdir('/Users/Pymard/Documents/Portfolio_Data/db/Credit')
#print(os.listdir('/Users/Pymard/Documents/Portfolio_Data/db/Credit'))

#############################################################################################

def main():

	train = load_data()

    
	page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Data Exploration', 'Prediction', 'Analytics'])

	if page == 'Homepage':
		st.title(':moneybag:Analytics and Machine Learning App')
		st.markdown('👈Select a page in the sidebar')
		st.markdown('This application performs machine learning predictions on loan applications and outputs predictions of approved or rejected applications.')

		st.markdown('This application provides:')
		st.markdown('●    A machine learning prediction on loan applications.:computer:')
		st.markdown('●    Data Exploration of the dataset used in training and prediction.:bar_chart:')
		st.markdown('●    Custom data Visualization and Plots.:chart_with_upwards_trend:')

		if st.checkbox('Show raw Data'):
			st.dataframe(train)
	elif page == 'Data Exploration':
		st.title('Explore the Dataset')
		if st.checkbox('Show raw Data'):
			st.dataframe(train)
        
		st.markdown('### Analysing column distribution')
		all_columns_names = train.columns.tolist()
		selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
		if st.button("Generate Plot"):
			st.success("Generating Customizable Bar Plot for {}".format(selected_columns_names))
			cust_data = train[selected_columns_names]
			st.bar_chart(cust_data)

		#if st.checkbox("Generate Pie Plot"):
			#all_columns = train.columns.to_list()
			#column_to_plot = st.selectbox("Select 1 Column",all_columns)
			#pie_plot = train[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
			#st.write(pie_plot)
			#st.pyplot()

		if st.checkbox("Show Shape"):
				st.write(train.shape)

		if st.checkbox("Show Columns"):
				all_columns = train.columns.to_list()
				st.write(all_columns)

		if st.checkbox("Summary"):
				st.write(train.describe())

		if st.checkbox("Show Selected Columns"):
				all_columns = train.columns.to_list()
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df = train[selected_columns]
				st.dataframe(new_df)

		if st.checkbox("Show Value Counts"):
				st.write(train.iloc[:,0].value_counts())

		if st.checkbox("Correlation Plot(Matplotlib)"):
				plt.matshow(train.corr())
				st.pyplot()

		if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(train.corr(),annot=True))
				st.pyplot()
        

	elif page == 'Analytics':
		st.title('Analytics.:bar_chart:')
		st.markdown('Upload your custom dataset and run visualisations. The downloaded batch prediction file can also be uploaded here so as to run visualisations on loan statuses.')
		uploaded_file = st.file_uploader("Upload a Dataset (CSV) for Analysis", type="csv")
		if uploaded_file is not None:
			df = pd.read_csv(uploaded_file)
			if st.checkbox('Show data'):
				st.dataframe(df)
			# Customizable Plot
			pd.set_option('plotting.backend', 'pandas_bokeh')
			st.subheader("Customizable Plot")
			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line", "hist","step","point", "scatter", "barh","map"])
			selected_columns_names = st.multiselect("Select Column(s) To Plot",all_columns_names)

			if st.button("Generate Plot"):
				st.success("Generating Customizable {} Plot of {}".format(type_of_plot,selected_columns_names))

				#Custom Plots
				if type_of_plot == 'scatter':
					if len(selected_columns_names) == 1:
						cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
						st.bokeh_chart(cust_plot)
					else:
        					cust_plot = df[selected_columns_names].iloc[:, [0,1]].plot.scatter(x = df[selected_columns_names].columns[0], y = df[selected_columns_names].columns[1])
        					st.bokeh_chart(cust_plot)          			        	
        		

				#Custom Plot Pie
				elif type_of_plot == 'pie2':
					cust_data = df[selected_columns_names].plot.pie(y = df[selected_columns_names].columns[0])
					st.bokeh_chart(cust_data)


				# Custom Plot 
				elif type_of_plot:
					cust_plot2= df[selected_columns_names].plot(kind=type_of_plot)
					st.bokeh_chart(cust_plot2)  #st.write(cust_plot)
					#st.pyplot()

				
                         


       
 
	else:
		st.title('Modelling')
		st.markdown('Input values in the form below for predictions or upload a csv file in the side menu for batch predictions.')
		model, accuracy = train_model(train)
		#st.write('Accuracy: ' + str(accuracy))
		Current_Loan_Amount = st.number_input("Enter Loan Amount", 0, 90000000, 0)
		Credit_Score = st.number_input("Credit Score", 300, 1255, 300)
		Annual_Income =  st.number_input("Annual Income", 0, 9000000, 0)
		Years_in_current_job = st.number_input("Enter Years in Current Job", 0, 20, 0)
		Term_Short_Term	 = st.selectbox("Loan Tenure", ['Short Term','Long Term'])
		Home_Ownership_Home_Mortgage = st.selectbox('Mortgage?',["Yes", "No"])
		Home_Ownership_Own_Home = st.selectbox('Own Home?',["Yes", "No"])
		Home_Ownership_Rent = st.selectbox('Rent?',["Yes", "No"])
		uploaded_file = st.sidebar.file_uploader("Upload a CSV file for Batch Prediction", type="csv")
		if uploaded_file is not None:
			data = pd.read_csv(uploaded_file)
			data2 = data.copy()
			data = pd.get_dummies(data, drop_first = True)
			#st.dataframe(data.head())
			data = data.values
			prediction = model.predict(data)
			if st.sidebar.button("Prediction"):
				submit = data2
				submit['Loan_Status'] = prediction
				submit['Loan_Status'] = submit['Loan_Status'].map({1: 'Approved', 0: 'Rejected'})
				st.sidebar.info('Batch Prediction Completed!')
				def get_table_download_link(df):
				
					csv = df.to_csv(index=False)
					b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
					href = f'<a href="data:file/csv;base64,{b64}" download="Predictions.csv">Download csv file</a>'
					return href
				st.sidebar.markdown(get_table_download_link(submit), unsafe_allow_html=True)
				
				
        		
				


		P = [[Current_Loan_Amount,Credit_Score,Annual_Income,Years_in_current_job,Term_Short_Term,Home_Ownership_Home_Mortgage, Home_Ownership_Own_Home,Home_Ownership_Rent ]]

		P = pd.DataFrame(columns=['Current_Loan_Amount','Credit_Score','Annual_Income','Years_in_current_job',
                          'Term_Short_Term','Home_Ownership_Home_Mortgage', 'Home_Ownership_Own_Home', 'Home_Ownership_Rent'], data = P)
		#st.dataframe(P)

		P = pd.get_dummies(data = P , columns = ['Term_Short_Term','Home_Ownership_Home_Mortgage', 'Home_Ownership_Own_Home', 'Home_Ownership_Rent'])
		#st.dataframe(P)

		predictions = model.predict(P)



		if st.button("Predict"):
			result = predictions
			if result[0] == 0:
				Predict = 'Loan Application is Rejected'
	  
			else:
				result[0] == 1
				Predict = 'Loan Application is Approved'

			st.success("{}".format(Predict))

##############################################################################################################


@st.cache(allow_output_mutation=True)
def train_model(train):

	le = LabelEncoder()
	cols = ['Term', 'Home Ownership']
	train['Loan Status'] = le.fit_transform(train['Loan Status'])

	train = pd.get_dummies(data = train, columns = cols, drop_first = True)

	X = train.drop(columns = ['Purpose','Monthly Debt','Years of Credit History',
                         'Number of Open Accounts', 'Number of Credit Problems', 'Current Credit Balance',
                         'Maximum Open Credit','Bankruptcies','Tax Liens'])
	y = train['Loan Status']

	from imblearn.over_sampling import RandomOverSampler
	ros = RandomOverSampler()
	X_ros, y_ros = ros.fit_sample(X, y)
	#print(X_ros.shape[0] - X.shape[0], 'new random picked points')

	y = X_ros['Loan Status'].values
	X = X_ros.drop(columns = ['Loan Status']).values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	model = ExtraTreesClassifier()
	model.fit(X_train, y_train)
	pred = model.predict(X_test)

	return model, model.score(X_test, y_test)

##################################################################################################



@st.cache(allow_output_mutation=True)
def load_data():
	train = pd.read_csv('credit_train.csv')

	train.drop_duplicates(keep=False,inplace=True)

	#Data Cleaning

	train['Years in current job'] = train['Years in current job'].replace('10+ years', '10')
	train['Years in current job'] = train['Years in current job'].replace('8 years', '8')
	train['Years in current job'] = train['Years in current job'].replace('5 years', '5')
	train['Years in current job'] = train['Years in current job'].replace('< 1 year', '1')
	train['Years in current job'] = train['Years in current job'].replace('1 year', '1')
	train['Years in current job'] = train['Years in current job'].replace('6 years', '6')
	train['Years in current job'] = train['Years in current job'].replace('9 years', '9')
	train['Years in current job'] = train['Years in current job'].replace('2 years', '2')
	train['Years in current job'] = train['Years in current job'].replace('3 years', '3')
	train['Years in current job'] = train['Years in current job'].replace('4 years', '4')
	train['Years in current job'] = train['Years in current job'].replace('7 years', '7')

	Q1 = train.quantile(0.25)
	Q3 = train.quantile(0.75)
	IQR = Q3 - Q1

	train = train[~((train < (Q1 - 0.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
	#train.shape

	train['Years in current job'] = pd.to_numeric(train['Years in current job'])
	train.drop_duplicates('Loan ID', inplace = True)

	train = train.drop(columns = ['Months since last delinquent'])
	train['Credit Score'] = train['Credit Score'].fillna(train['Credit Score'].mean())
	train['Annual Income'] = train['Annual Income'].fillna(train['Annual Income'].mean())
	train['Years in current job'] = train['Years in current job'].fillna(train['Years in current job'].mean())
	train = train.drop(columns = ['Loan ID','Customer ID'])
	return train



if __name__ == '__main__':
    main()