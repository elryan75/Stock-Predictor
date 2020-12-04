from flask import Flask, url_for, render_template, request, redirect
import forecast
import stockInfo


app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def index():
    if request.method == "POST":
        ticker = request.form["nm"]
        stock = stockInfo.getInfo(ticker)
        prediction = forecast.getPrediction(ticker, stock)
        if(prediction[0] >= 0):
            color = "green"
        else:
            color = "red"
        return render_template("index.html", stock=prediction[0], ticker=ticker, color=color, image=prediction[1])
    else:
        return render_template("index.html", stock="", image=None)







if __name__ == '__main__':
    app.run(port=4996, debug=True)