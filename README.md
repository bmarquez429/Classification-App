# Classification App

The Python scripts in this branch generate a version of the app that allows for the saving of one or more binary classifiers in joblib or pickle file formats. When the app is deployed in a local machine, the files are saved in a Downloads folder of the machine. When the app is deployed in [Streamlit Community Cloud](https://streamlit.io/cloud), the files are saved in a Downloads folder in the cloud. The latter scenario appears to be a limitation of the app since it is not clear how to access the saved files in the cloud.
