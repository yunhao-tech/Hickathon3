# Week 1

## Part 0: Plotly
`Plotly`'s Python graphing library makes interactive, publication-quality graphs. Here are [documentations](https://plotly.com/python/). There are really "all" kinds of figures.

With Dash, we can build machine learning web apps in Python, such as NLP translation web app, GAN demos, regression model. It is very helpful with a good visualization tool. Demos are [here](https://plotly.com/building-machine-learning-web-apps-in-python/)




## Part 1: Data cleaning 

- File format: `Feather`. It is a binary file and it takes less memory space and less time when we do I/O operations. Supported by `pandas`. 

- package `missingno`: a small toolset for **missing data visualizations**. Allows us to get a quick visual summary of the completeness our dataset.

- `isna()` and `.duplicated()` to detect the Nan and duplicate. A pratical code style: (study its naming habitude and logic, name the boolean list as `mask` to have a better understanding)

```python
mask_lat_lng_na = (
    building_meta.lat.isna().values
    | building_meta.lng.isna().values
) # | means 'or'

building_meta = building_meta[~mask_lat_lng_na].reset_index(drop=True)
# ~ means 'not'
```

- `.info()` for dataframe shows the count and dtypes of data; `.describe()` for dataframe shows the basic statistics of attributes.

- Combine plt and `seaborn` to get beautiful figures

```python
sns.set(rc={'figure.figsize': (24, 12)})
sns.set(font_scale=1.5)
f, axes = plt.subplots(2, 3)
axes = axes.flatten()
color = "dodgerblue"

ax1 = axes[0]
g1 = sns.distplot(weather["air_temperature"].dropna(), ax=ax1, color=color)
ax1.set_title('Air temperature (ºC)')
ax1.set(xlabel="")

...
# Remove empty axes
f.delaxes(axes[5])

plt.tight_layout()
plt.show()
```
To notice: how to combine `plt` and `seaborn`; how to use `.flatten()`; choose a beautiful color; how to **drop empty axes**.

## Part 2: data visualisation
- use of `pivot_table` to better analyse data. It is similar to `groupby`. Use case in detail: [zhihu](https://zhuanlan.zhihu.com/p/31952948)

- use `heatmap` in `seaborn` to visualize the relationship between variables. Use case on [zhihu](https://zhuanlan.zhihu.com/p/96040773). We can use a `pivot_table` as input.

- sometimes, think to pass to `log-scale` to better visualize.

- **Code style**: we can change line for each call of function
```python
_ = (
    meters
    .set_index('timestamp')
    .resample("D")
    .meter_reading
    .mean()
    .plot()
    .set_ylabel('Mean meter reading', fontsize=13)
)
```

## Part 3: Feature Engineering
Some ideas of Feature Engineering:

- For timestamps, we can extract some information to form a new feature, such as **hour, weekday, day, month, year, is_wider_busness_hours, is_weekend, season...**. It really depends on concret application. Example:
```python
# days of the week (mon=0 and sun=6)
data["weekday"] = data["timestamp"].dt.dayofweek.astype("int8")

# month
data["month"] = data["timestamp"].dt.month.astype("int8")

# year
data["year"] = data["timestamp"].dt.year.astype("int16")

# business hours
data['is_wider_busness_hours'] = np.where((data["hour"] >= 7) & (data["hour"] <=19 ), 1, 0)

# Weekend
data['is_weekend'] = np.where((data["weekday"] >= 0) & (data["weekday"] <= 4), 0, 1)

# Season of year
data['season'] = (np.where(data["month"].isin([12, 1, 2]), 0,
                   np.where(data["month"].isin([3, 4, 5]), 2,         
                   np.where(data["month"].isin([6, 7, 8]), 3,          
                   np.where(data["month"].isin([9, 10, 11]), 1, 0)))))
```

- For geographic position, maybe it is a good idea to divide into different regions (province, country, continent...)

- For normal numerical features, we can add statistical features (mean, sum, `log` or `ratio`). Also, we can transform a numerical feature into a `categorical feature` by cutting it into classes.

# Week 2: Modeling
This week, the classical machine learning methods are introduced, including linear models, tree-based methods (random forests, Gradient Boosting, XGBoost), LDA, etc...

Serveral key points to notice:

- For linear model, we can plot coefficient importance; for other methods, we can plot the `feature importance`, which is useful for feature selection. The function `plot_importance` in `utils.py` can do this.

- `stacking of models`: instead of using trivial functions (such as the mean) to aggregate the predictions of all predictors in a set in the case of a random forest for instance, why not **train a model to perform this aggregation**?

Let' assume that we have B differents models represented by $\hat M_1,\hat M_2, \hat M_3,... \hat M_B $. 

We divide $X_{train}$ in two parts : $(X_{train_1}, X_{train_2})$.

We train $\hat M_1,\hat M_2, \hat M_3,... \hat M_B $ on $X_{train_{1}}$.

We make B differents predictions on $X_{train_2}$ : $ \hat y_1 , \hat y_2, ..., \hat y_B$

If we denote $\hat X =  (\hat y_1 , \hat y_2, ..., \hat y_B)$, Then we train a last model $\hat M_L$ on $(y_{train_2} , \hat X )$ 

Finally several models are trained on the data and a **last model is trained on the predictions of the other models**. This is a new training set! This last model is called a `meta model`. We can choose a linear model because we find that it makes sense to model the predictions of the others with a linear model; however it's not necessarily the best choice.

Sklearn has a function `StackingRegressor` to do stacking of models.

- Classification metrics: accuracy, precision, recall, F1-score.

`accuracy`: what percentage of my predictions are correct?

`precision`: if I focus on a given class (say, fraudulent payments): **what percentage of predicted frauds are actual frauds**?

`recall`: if I focus on a given class (say, fraudulent payments): **what percentage of the actual frauds can I correctly identify** with my model?

`f1-score` has no simple direct interpretation, but it's a handy measure when you want to monitor both precision and recall and summarize them in a single metric. There is a tradeoff between precision and recall. We can modify the threshold of classification to search a best f1-score.

There is one paragraph which analyse these metrics on a true dataset. We can borrow this analysis in other practical cases. The `balance of the classes` is a key factor to take into account when we analyse the result.

- Confusion matrix is another helpful tool to analyse classification result. We can find function `ConfusionMatrixDisplay` in sklearn.

- When spliting the dataset, we should take into account the balance of classes, by setting the argument `stratify` in function `train_test_split`. In this way, for each class, there would be 20% data in test set.
```{python}
Xtrain, Xval, Ytrain, Yval = train_test_split(data, target, test_size=0.2, stratify=target)
```