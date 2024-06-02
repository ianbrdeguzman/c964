import pandas as pd
import pickle
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Annotated
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# db path
db_path = "./database/wine.db"
# create sqlite3 connection
cx = sqlite3.connect(db_path)
# get cursor
cu = cx.cursor()
# create prediction table if not exist
# cu.execute("DROP TABLE IF EXISTS prediction;")
cu.executescript(
    """
    CREATE TABLE IF NOT EXISTS prediction (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      fixed_acidity REAL,
      volatile_acidity REAL,
      citric_acid REAL,
      residual_sugar REAL,
      chlorides REAL,
      free_sulfur_dioxide REAL,
      total_sulfur_dioxide REAL,
      density REAL,
      ph REAL,
      sulphates REAL,
      alcohol REAL,
      quality REAL,
      date NUMERIC
    )
    """
)
# close connection
cx.close()

log_pipe_path = "./model/log_pipe.pickle"
log_pipe_file = Path(log_pipe_path)

# load csv file
df = pd.read_csv(open("./data/winequality-red.csv"))

# plot alcohol vs quality
image_path = "./static/quality_vs_alcohol.png"
# sns.boxplot(x="quality", y="alcohol", data=df)
# plt.savefig(image_path, dpi=400)

# create model if not exist
if not log_pipe_file.exists():
    # split independent and dependent df values
    y = df.values[:, 11]
    X = df.values[:, 0:11]

    # split training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    # initialize pipeline
    log = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])

    # train log model
    log.fit(X_train, y_train)

    # make prediction
    predictions = log.predict(X_test)

    # mean square
    mse = mean_squared_error(predictions, y_test)

    print("MSE", mse)

    # serialize trained model
    pickle.dump(log, open(log_pipe_path, "wb"))

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# function to get top predictions
def get_top_predictions():
    top_predictions = []

    cx = sqlite3.connect(db_path)
    cu = cx.cursor()
    for row in cu.execute("SELECT * FROM prediction ORDER BY quality DESC LIMIT 5;"):
        top_predictions.append(row)
    cx.close()

    return top_predictions


# function to get recent predictions
def get_recent_predictions():
    recent_predictions = []

    cx = sqlite3.connect(db_path)
    cu = cx.cursor()
    for row in cu.execute("SELECT * FROM prediction ORDER BY date DESC LIMIT 5;"):
        recent_predictions.append(row)
    cx.close()

    return recent_predictions


# function to make a prediction
def make_prediction(
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    ph,
    sulphates,
    alcohol,
):
    # load model
    log_pipe = pickle.load(open(log_pipe_path, "rb"))

    # make a prediction
    [prediction] = log_pipe.predict(
        [
            [
                fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                ph,
                sulphates,
                alcohol,
            ]
        ]
    )

    return prediction


# function to save prediction
def save_prediction(
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    ph,
    sulphates,
    alcohol,
    prediction,
):
    cx = sqlite3.connect(db_path)
    cu = cx.cursor()
    cu.execute(
        """
        INSERT INTO prediction (
          fixed_acidity,
          volatile_acidity,
          citric_acid,
          residual_sugar,
          chlorides,
          free_sulfur_dioxide,
          total_sulfur_dioxide,
          density,
          ph,
          sulphates,
          alcohol,
          quality,
          date
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            ph,
            sulphates,
            alcohol,
            prediction,
            datetime.today(),
        ),
    )
    cx.commit()
    cx.close()


@app.get("/")
def get_home(request: Request):
    top_predictions = get_top_predictions()
    recent_prediction = get_recent_predictions()

    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "top_predictions": top_predictions,
            "recent_predictions": recent_prediction,
            "src": image_path,
        },
    )


@app.post("/")
def post_home(
    request: Request,
    fixed_acidity: Annotated[float, Form()],
    volatile_acidity: Annotated[float, Form()],
    citric_acid: Annotated[float, Form()],
    residual_sugar: Annotated[float, Form()],
    chlorides: Annotated[float, Form()],
    free_sulfur_dioxide: Annotated[float, Form()],
    total_sulfur_dioxide: Annotated[float, Form()],
    density: Annotated[float, Form()],
    ph: Annotated[float, Form()],
    sulphates: Annotated[float, Form()],
    alcohol: Annotated[float, Form()],
):
    prediction = make_prediction(
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol,
    )

    save_prediction(
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol,
        prediction,
    )

    # get top predictions
    top_predictions = get_top_predictions()
    recent_predictions = get_recent_predictions()

    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "fixed_acidity": fixed_acidity,
            "volatile_acidity": volatile_acidity,
            "citric_acid": citric_acid,
            "residual_sugar": residual_sugar,
            "chlorides": chlorides,
            "free_sulfur_dioxide": free_sulfur_dioxide,
            "total_sulfur_dioxide": total_sulfur_dioxide,
            "density": density,
            "ph": ph,
            "sulphates": sulphates,
            "alcohol": alcohol,
            "prediction": prediction,
            "top_predictions": top_predictions,
            "recent_predictions": recent_predictions,
        },
    )
