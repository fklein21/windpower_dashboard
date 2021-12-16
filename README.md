# Windpower Dashboard

This dashboard was developed as part of the capstone project about wind power predictions. It is used to visualize the outputs of the data analysis and modeling work.

To explore the dashboard go to the live version on Heroku: [Energy Output Forecast for the Next 24 Hours](https://windpower-forecast.herokuapp.com)

For more information about the project visit the [project page on GitHub](https://github.com/JeromeSauer/Capstone_WindPowerPredicting).



## Requirements:

This dashboard uses `Python v3.9.9`. \ 

You can install it with
 ```console
 $ pyenv install 3.8.11
 ```

Setup the environment with:

```console
$ make setup
```

There are two requirement files in this template: 
1. **requirements_dev.txt** This is the requirements file you can use locally to set everything up and develop the dashboard. You can add as much here as you want to. 
2. **requirements.txt** This is the requirements that Heroku uses. Because on Heroku the memory for the dashboard is very limited it should not contain the development environment (e.g. jupyter) and as few libraries as possible.


## How To Run Locally

1. Install `pyenv`:
```console
$ brew install pyenv
```

2. Open a terminal in the project root directory and run:
```console
$ make setup
```

3. Then run the command:
```console
$ source .venv/bin/activate
```

4. Then install the dependencies:
```console
$ pip install -r requirements.txt
```

5. Finally start the web server:
```console
$ python dashboard_main.py
```

This server will start on port 8050 by default. You can change this in `app.py` by changing the following line to this:

```python
if __name__ == "__main__":
    app.run_server(debug=True, port=<desired port>)
```

## How to deploy on Heroku

This dashboard is hosted on Heroku.
The workflow to deploy it on Heroku is detailed below.

1. Develop and debug the dashboard locally.
2. Check the Heroku documantation if you're using the same python version as Heroku and make sure you have the python modules installed in the right versions.
3. Heroku is using another command to run the dashboard. Check if the dashboard is working locally with\
 ` $ gunicorn app:server`

5. Edit the Procfile that Heroku uses to start the dashboard. It should contain the following command:\
`web: gunicorn dashboard_main:server`

6. Deploy to Heroku
