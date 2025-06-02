



import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *

X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J
    
# For testing purposes:
movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)          #  Initialize my ratings


my_ratings[126]  = 5
my_ratings[151]  = 5
my_ratings[88]  = 5
my_ratings[250]  = 5
my_ratings[393]  = 5
my_ratings[523]  = 5
my_ratings[660]  = 5
my_ratings[666]  = 5
my_ratings[791]  = 5
my_ratings[793]  = 5
my_ratings[794]  = 5
my_ratings[899]  = 5
my_ratings[906]  = 5
my_ratings[1211]  = 5
my_ratings[1213]  = 5
my_ratings[1363]  = 5
my_ratings[1381]  = 5
my_ratings[1455]  = 5
my_ratings[1521]  = 5
my_ratings[1549]  = 5
my_ratings[1559]  = 5
my_ratings[1638]  = 5
my_ratings[1638]  = 5
my_ratings[1665]  = 5
my_ratings[1898]  = 5
my_ratings[1902]  = 5
my_ratings[1937]  = 5
my_ratings[1967]  = 5
my_ratings[1992]  = 5
my_ratings[2004]  = 5


print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}');

# Reload ratings and add new ratings
Y, R = load_ratings_small()
Y    = np.c_[my_ratings, Y]
R    = np.c_[(my_ratings != 0).astype(int), R]


# Normalize the Dataset
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 10 
 
# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)






Ynorm, Ymean = normalizeRatings(Y, R)


iterations = 200
lambda_ = 10
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
        



# For Testing Purposes:

# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_ratings:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')





# After training and making predictions

# 2. Add predictions and ratings to DataFrame
movieList_df["pred"] = my_predictions
movieList_df["my_rating"] = my_ratings  # Add your actual ratings

# 3. Convert indices to numpy array
ix_np = ix.numpy()

# 4. Create filters
popular_filter = (movieList_df["number of ratings"] > 10)  # Lowered threshold
unrated_filter = (movieList_df["my_rating"] == 0)  # Movies you haven't rated

# 5. Get top recommendations
top_recommendations = movieList_df.iloc[ix_np]  # Get full sorted list
filtered_recs = top_recommendations[popular_filter & unrated_filter]

# 6. If no popular movies, show top predictions regardless of popularity
if filtered_recs.empty:
    print("\nNo popular movies in top recommendations. Showing top overall predictions:")
    filtered_recs = top_recommendations[unrated_filter].head(20)
else:
    # Sort popular recommendations by mean rating
    filtered_recs = filtered_recs.sort_values("mean rating", ascending=False).head(20)

# 7. Print recommendations
print("\nTop Recommended Movies for You:")
print(filtered_recs[["pred", "mean rating", "number of ratings", "title"]].head(10))

# 8. Print model performance on rated movies
rated_movies = movieList_df[my_ratings > 0]
if not rated_movies.empty:
    print("\nModel Performance on Movies You Rated:")
    for i in range(len(rated_movies)):
        row = rated_movies.iloc[i]
        print(f'{row["title"]}: Actual {row["my_rating"]}, Predicted {row["pred"]:.2f}')
else:
    print("\nYou haven't rated any movies yet")




"""
#Add predictions to DataFrame
movieList_df["pred"] = my_predictions

# Reorganize columns with prediction first
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])

# Filter for popular movies (>20 ratings) and get top 300 predictions
filter = (movieList_df["number of ratings"] > 20)
recommendations = movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)

# ADD PRINT STATEMENT HERE
print("\nTop Recommended Movies (Popular & Highly Rated):")
print(recommendations.head(10))  # Show top 10 recommendations

"""
