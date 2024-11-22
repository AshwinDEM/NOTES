
Here, is a quick summary of data preprocessing and common machine learning commands
#### Preprocessing
### Dropping values
```python
df.dropna() # Drop all rows with NaN values in the dataset
df.dropna(subset = []) # Drop all rows with NaN values in colummns in subset
df.drop_duplicates() # Drop all duplicate rows
```

### Changing datatype
``` python
df['col'] = df['col'].astype(float) # Convert all values in col to float
df['col'] = pd.to_numeric(df['col'], errors = 'coerce') # Converts all values to numeric, if it cannot convert, it sets it as NaN
```

### Outlier Handling
Removing data points that are outside 3 standard deviations

```python
mean = df['col'].mean()
std = df['col'].std()
df = df[df['col'] >= mean - 3 * std & df['col'] <= mean + 3 * std]
```

### Filtering

```python
df_filtered = df[df['col'] > value] # Select rows that are greater than some value
df_filtered = df[['A', 'B']] # Selects only columns A and B
```

### Normalisation

Rescales the data to fit between -1 and 1
```python
df['col'] = (df['col'] - df['col'].min())/(df['col'].max() - df['col'].min())
```
In a easier to see way

$$
df[col] = \frac{df[col] - df[col].min()}{df[col].max() - df[col].min()}
$$

### Standardisation

Data has a mean of 0 and standard deviation of 1.
```python
df['col'] = (df['col'] - df['col'].mean())/df['col'].std()
```

### Feature Encoding
Converting categorical variables to numerical format

1. Label Encoder assigns a unique value to each category. 
```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['col'] = label_encoder.fit_transform(df['col'])
```
2. One Hot Encoder makes a new column for each categorical value in the column, and if a column in 1, then the value for that category is that column name
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
df['col'] = encoder.fit_transform(df['col'])
```

### Dimensionality Reduction
Reduces the number of features, but stores the important information.

1. **Principal component analysis** projects the data into a lower dimension
```python
from sklearn.decomposition import PCA

pca = PCA(n_components = 2) # 2 as an example
df_pca = pca.fit_transform(df)
```
2. **Linear Discriminant Analysis** maximises class separability
```python
from sklearn.disciminant_analysis import LinearDiscriminatAnalysis as LDA

lda = LDA(n_components = 2)
df_lda = lda.fit_transform(df)
```

3. **t-Distributed Stochastic Neighbor Embedding** for visualising higher dimensions
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components = 2)
df_tsne = tsne.fit_transform(df)
```
