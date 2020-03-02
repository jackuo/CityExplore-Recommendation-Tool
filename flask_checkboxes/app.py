from flask import Flask, render_template, request

app = Flask(__name__)


def return_something(checked_list):
    return_text = "The indicies are: ", checked_list
    return flask.render_template('return.html', return_text)


@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST': #imports the request object - what was posted in front end
        print(request.form.getlist('mycheckbox'))
        return 'Done'
    return render_template('index.html')

@app.route('/return_something', methods=['GET', 'POST'])
def ranking():
	checked_list = request.form.getlist('mycheckbox')
	return render_template(return_something(checked_list)
