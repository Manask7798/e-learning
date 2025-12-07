from flask import Flask, render_template, request, make_response

app = Flask(__name__, static_url_path="/static")


@app.route('/')
def home():
    begriff = request.cookies.get('einheit')
    return render_template('course.html', einheit=begriff)


@app.route('/about')
def about():
    return render_template('about-us.html')


@app.route('/begriff')
def begriff():
    resp = make_response(render_template('begriff.html'))
    resp.set_cookie('einheit', 'true')
    return resp


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/impress')
def impress():
    return render_template('impress.html')


@app.route('/lernerfolge')
def lernerfolge():
    return render_template('lernerfolge.html')


@app.route('/massnahmen')
def massnahmen():
    return render_template('massnahmen.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


if __name__ == '__main__':
    app.run(debug=True)
