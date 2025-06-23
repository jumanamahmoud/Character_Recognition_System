# Character_Recognition_System

This is a system designed for the Computational Intelligence course by Group 11.
Group Members:
Jomana Mahmoud A22MJ3005
Tansim Jannat A22MJ3012
Jayotshna Sengeny A22MJ8015
Nadeeya Binti Azizee A22MJ8001

# Details
It uses the MJSynth dataset imported from https://huggingface.co/datasets/priyank-m/MJSynth_text_recognition to detect characters in an image.
It then converts those characters into voice. This repository does not include the dataset folder. The folder will be manually installed using code in the Main.ipynb file.

# Instructions:
1. Install Streamlit in your terminal via "pip install streamlit" to run the "app.py" file. (or pip3)
2. Open JupyterLab "py -m jupyterlab" (if "py" doesn't work, try "python" or "python3")
3. Run the code (Main.ipynb) cell by cell to correctly install and load the dataset.
4. The code also trains the model and saves its state/weights in the crnn_model.pth pytorch file for usage later.
5. Run "streamlit run app.py" in the correct directory on a new terminal page to open the webpage.
6. You can now upload images to the system.
Note: This repo does not have the old system (HDRS).


# Citations for the dataset:

@InProceedings{Jaderberg14c, author = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman", title = "Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition", booktitle = "Workshop on Deep Learning, NIPS", year = "2014", }

@Article{Jaderberg16, author = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman", title = "Reading Text in the Wild with Convolutional Neural Networks", journal = "International Journal of Computer Vision", number = "1", volume = "116", pages = "1--20", month = "jan", year = "2016", }
