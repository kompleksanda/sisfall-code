instructions oh how to execute the project
Note that this is a combination of the test and train codes plus the web application codes

To run the application

1) Download the SisFall dataset and place it in the visualizer directory inside the project folder.
    Make sure the visualizer directory structure looks like this after extraction.

    visualizer
    ...........SisFall_dataset
                        ....SA01
                        ....SA02
    ...

2)  With the anaconda command prompt
    cd into the directory that has this README file you're reading then type

    python manage.py runserver

3) open your web browser and type this link:

   http://127.0.0.1:8000/dashboard/

To train the dataset

4) In an anaconda command prompt, type
    python sisfall_svd.py
5) Next, type:
    python shape_info.py
6) train the model by typing
    python model_cnn.py
7) Lastly draw the bar chart with
    python draw.py