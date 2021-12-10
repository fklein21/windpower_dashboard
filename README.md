# Windpower_Dashboard

## Workflow for development

1. Set up the environment
2. Work in notebook to get acquainted with dash plots
3. Work in app.py-File to create a dash
4. Check if Dashboard is working locally with

- ` $ python app.py`
- ` $ gunicorn app:server`

6. Deploy to Heroku

























# FlaskIntroduction

This repo has been updated to work with `Python v3.8` and up.

### How To Run
1. Install `virtualenv`:
```
$ pip install virtualenv
```

2. Open a terminal in the project root directory and run:
```
$ virtualenv env
```

3. Then run the command:
```
$ .\env\Scripts\activate
```

4. Then install the dependencies:
```
$ (env) pip install -r requirements.txt
```

5. Finally start the web server:
```
$ (env) python app.py
```

This server will start on port 5000 by default. You can change this in `app.py` by changing the following line to this:

```python
if __name__ == "__main__":
    app.run(debug=True, port=<desired port>)
```