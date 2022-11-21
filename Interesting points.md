# Part 0: Plotly
`Plotly`'s Python graphing library makes interactive, publication-quality graphs. Here are [documentations](https://plotly.com/python/). There are really "all" kinds of figures.

With Dash, we can build machine learning web apps in Python, such as NLP translation web app, GAN demos, regression model. It is very helpful with a good visualization tool. Demos are [here](https://plotly.com/building-machine-learning-web-apps-in-python/)




# Part 1: Data cleaning 

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

# Part 2: data visualisation
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

# Part 3: Feature Engineering
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