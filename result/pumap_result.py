from flask import Flask, render_template
# from ..umap import umap_panene
# from ..utils import load_merge_mnist, load_merge_cifar, draw_plot

import pandas as pd

app = Flask(__name__, template_folder='', static_folder='')

@app.route('/')
def result():

    # x, y = load_merge_mnist()
    # item = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    # embedding = umap_panene.UMAP(n_neighbors=5).fit_transform(X=x, y=None, label=y, item=item, progressive=True)
    # draw_plot(embedding, y, item, "myimage")
    # print("image saved")

    ddd = pd.read_csv("result/fashion/y.csv")
    

    return render_template('pumap_result.html', data=ddd)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) # FLASK_APP=app.py python -m flask run