# Character_Recognition_System

This is a system designed for the Computational Intelligence course. 
It uses the MJSynth dataset imported from https://huggingface.co/datasets/priyank-m/MJSynth_text_recognition to detect characters in an image.
It then converts those characters into voice.

# Instructions:
1. Install streamlit in your terminal via "pip install streamlit" in order to run the "app.py" file.
2. Open JupyterLab "py -m jupyterlab" (if "py" doesn't work, try "python" or "python3")
3. Run the code (Updated_Main) cell by cell to correctly install and load the dataset.
4. Run "streamlit run app.py" in the correct directory on a new terminal page to open the webpage.
5. To make any small changes to the code (UI only, please), just click on the app.py file on GitHub and access editor mode from top right corner to quickly make changes.
Note: This repo does not have the old system (HDRS); to access that, you need to go back to the old repo.




Citation details provided on the source website (if you use the data, please cite):

@InProceedings{Jaderberg14c, author = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman", title = "Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition", booktitle = "Workshop on Deep Learning, NIPS", year = "2014", }

@Article{Jaderberg16, author = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman", title = "Reading Text in the Wild with Convolutional Neural Networks", journal = "International Journal of Computer Vision", number = "1", volume = "116", pages = "1--20", month = "jan", year = "2016", }
