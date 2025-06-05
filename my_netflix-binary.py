



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

def transform_ratings(arr):
    # Create a copy to avoid modifying original array
    result = np.copy(arr)

    # Get 0.5 "out of the way"
    result[(result == 0.5)] = 1
    
    # Replace 0 with 0.5
    result[(result == 0)] = 0.5
  
    # Replace 1 and 2 with 0
    result[(result > 0.5) & (result < 3)] = 0
    
    # Replace 3,4,5 with 1
    result[result >= 3] = 1
    
    return result




    
def count_ones(arr):
    """Counts all occurrences of the integer 1 in a NumPy array"""
    return int(np.count_nonzero(arr == 1))
    
def count_zeros(arr):
    """Counts all occurrences of the integer 1 in a NumPy array"""
    return int(np.count_nonzero(arr == 0))
    
def has_invalid_entries(arr):
    """Returns True if array contains values other than -1, 0, or 1"""
    # Create a mask of values outside [-1, 0, 1]
    invalid_mask = (arr != 0) & (arr != 0.5) & (arr != 1)
    return np.any(invalid_mask)
    

    
binaryY = transform_ratings(Y)       
ones = count_ones(binaryY)
print("ones ", ones)
zeros = count_zeros(binaryY)
print("zeros ", zeros)
invalid = has_invalid_entries(binaryY)
print("invalid ", invalid)


# Create training and testing dataset
def decompose_matrix_80_20(R, random_seed=None):
    R = np.array(R)
    A = np.zeros_like(R)
    B = np.zeros_like(R)

    if random_seed is not None:
        np.random.seed(random_seed)

    count_ones = 0

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] == 1:
                # Assign to A with 80% chance, else to B
                if np.random.rand() < 0.8:
                    A[i, j] = 1
                else:
                    B[i, j] = 1
                count_ones += 1

    return A, B

def replace_with_half(Y, R):
    # Ensure both are NumPy arrays
    X = np.array(Y)
    R = np.array(R)

    # Validate shape compatibility
    if X.shape != R.shape:
        raise ValueError("Matrices Y and R must have the same shape.")

    # Create a copy of X to avoid modifying original
    result = Y.copy()

    # Replace entries in X with 0.5 wherever R is 0
    result[R == 0] = 0.5

    return result


# Testing
A,B = decompose_matrix_80_20(R)
Y_train = replace_with_half(binaryY,A)
Y_test = replace_with_half(binaryY,B)

#Test:
print("make sure they add to original matrix R")
print(np.all(A + B == R))

# This is just for convenience
new_train = Y_train.copy()
new_test = Y_test.copy()
new_original = binaryY.copy()

# Replace 0s with -1
new_train[Y_train == 0] = -1
new_test[Y_test == 0] = -1
new_original[binaryY == 0] = -1

# Replace 0.5s with 0
new_train[Y_train == 0.5] = 0
new_test[Y_test == 0.5] = 0
new_original[binaryY == 0.5] = 0

print("make sure the actual recommendations add up to the original")

print(np.all(new_train + new_test == new_original))

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Binary classification cost function for collaborative filtering.
    Uses sigmoid cross-entropy loss for binary ratings (0 = dislike, 1 = like).
    Unrated items are encoded as 0.5 and ignored in loss.

    Args:
      X (tf.Tensor): Item feature matrix (num_movies x num_features)
      W (tf.Tensor): User parameter matrix (num_users x num_features)
      b (tf.Tensor): User bias vector (1 x num_users)
      Y (tf.Tensor): Rating matrix (num_movies x num_users), values: 0, 1, 0.5
      R (tf.Tensor): Mask matrix (1 if rated, 0 if unrated)
      lambda_ (float): Regularization parameter

    Returns:
      total_cost (float): Total cost
    """
    # Create a mask matrix R where R[i,j] = 1 if the item was rated (Y[i,j] == 0 or 1)
    R = tf.where(tf.equal(Y, 0.5), tf.constant(0.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32))  # 1 for rated, 0 for unrated

    # Convert Y to float32 for TensorFlow compatibility
    Y = tf.cast(Y, tf.float32)

    # Ensure Y contains only 0 or 1 in rated positions (for binary cross-entropy)
    Y_binary = tf.where(tf.equal(R, 1), Y, tf.zeros_like(Y))

    # Compute predicted logits: X @ W^T + b
    logits = tf.matmul(X, W, transpose_b=True) + b

    # Compute binary cross-entropy loss between predicted logits and binary labels
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_binary, logits=logits)

    #This is equivalent to:
    #sig_pred = 1 / (1 + tf.exp(-logits))
    #loss = -Y_binary * tf.math.log(sig_pred) - (1 - Y_binary) * tf.math.log(1 - sig_pred)
  
    # Apply mask to ignore unrated items (where R == 0)
    masked_loss = loss * R

# Step 5: Compute total cost with regularization
    total_cost = tf.reduce_sum(masked_loss) + (lambda_ / 2) * (tf.reduce_sum(X**2) +     tf.reduce_sum(W**2))

    return total_cost

#Numerical Stability : Avoid large logits by initializing X and W with small random values.


def predict(X, W, b, user_idx, item_idx):
    logits = X[item_idx] @ W[user_idx] + b[0, user_idx]
    probability = tf.sigmoid(logits).numpy()
    return 1 if probability >= 0.5 else 0

# For testing purposes:
movieList, movieList_df = load_Movie_List_pd()


#  Useful Values
num_movies, num_users = Y.shape
num_features = 10 
 
# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float32),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float32),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float32),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
lambda_ = 1
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Y_train, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
          



def predict(X, W, b, user_idx, item_idx):
    logits = X[item_idx] @ W[user_idx] + b[0, user_idx]
    probability = tf.sigmoid(logits).numpy()
    return 1 if probability >= 0.5 else 0

def matrix_predict(X, W, b):
    """
    Predicts likes/dislikes for all user-item pairs using matrix operations.
    
    Args:
       W (np.ndarray or tf.Tensor): User parameter matrix (num_users x num_features)
        b (np.ndarray or tf.Tensor): User bias vector (1 x num_users)

    Returns:
        np.ndarray: Binary predictions (num_movies x num_users), values: 0 or 1
    """
    # Step 1: Compute logits = X @ W^T + b
    logits = tf.matmul(X, W, transpose_b=True) + b

    # Step 2: Apply sigmoid to get probabilities
    probabilities = tf.sigmoid(logits)

    # Step 3: Threshold at 0.5 to get binary predictions
    binary_predictions = (probabilities >= 0.5).numpy().astype(int)

    return binary_predictions

# Make a prediction using trained weights and biases
P = matrix_predict(X, W, b)

def compute_accuracy(P, Y_test, R):
    # Ensure inputs are NumPy arrays
    P = np.array(P)
    Y_test = np.array(Y_test)
    R = np.array(R)

    # Validate shape compatibility
    if not (P.shape == Y_test.shape == R.shape):
        raise ValueError("All matrices must have the same shape.")

    # Get mask of non-zero entries in R
    mask = (R == 1)

    # Extract corresponding values from P and Y_test
    P_masked = P[mask]
    Y_masked = Y_test[mask]

    # Count matches where P == Y_test
    matches = np.sum(P_masked == Y_masked)

    # Total non-zero entries in R
    total = np.sum(mask)

    # Avoid division by zero
    accuracy = matches / total if total > 0 else 0.0

    return accuracy

print("accuracy on test set")
print(compute_accuracy(P,Y_test, B))





















'''



# For Testing Purposes:

my_ratings = np.zeros(num_movies) + 0.5       #  Initialize my ratings


my_ratings[126]  = 1
my_ratings[151]  = 1
my_ratings[88]  = 1
my_ratings[250]  = 1
my_ratings[393]  = 1
my_ratings[523]  = 1
my_ratings[660]  = 1
my_ratings[666]  = 1
my_ratings[791]  = 1
my_ratings[793]  = 1
my_ratings[794]  = 1
my_ratings[899]  = 1
my_ratings[906]  = 1
my_ratings[1211]  = 1
my_ratings[1213]  = 1
my_ratings[1363]  = 1
my_ratings[1381]  = 1
my_ratings[1455]  = 1
my_ratings[1521]  = 1
my_ratings[1549]  = 1
my_ratings[1559]  = 1
my_ratings[1638]  = 1
my_ratings[1638]  = 1
my_ratings[1665]  = 1
my_ratings[1898]  = 1
my_ratings[1902]  = 1
my_ratings[1937]  = 1
my_ratings[1967]  = 1
my_ratings[1992]  = 1
my_ratings[2004]  = 1

my_ratings[120]  = 0
my_ratings[150]  = 0
my_ratings[80]  = 0
my_ratings[258]  = 0
my_ratings[390]  = 0
my_ratings[520]  = 0
my_ratings[660]  = 0
my_ratings[660]  = 0
my_ratings[790]  = 0
my_ratings[790]  = 0
my_ratings[790]  = 0
my_ratings[890]  = 0
my_ratings[900]  = 0

print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0.5 :
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}');
for i in range(len(my_ratings)):
    if my_ratings[i] < 0.5 :
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}');

# Reload ratings and add new ratings
Y, R = load_ratings_small()
Y    = np.c_[my_ratings, Y]
R    = np.c_[(my_ratings != 0).astype(int), R]







my_predictions = p[:,0]

positive_predictions = my_predictions[my_predictions > 0.5]
print("positive predictions ", len(positive_predictions))
negative_predictions = my_predictions[my_predictions < 0.5]
print("negative predictions ", len(negative_predictions))

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0.5:
       print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')
for i in range(len(my_ratings)):
    if my_ratings[i] < 0.5:
       print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')        








def recommend_top_movies_with_quality(X, W, b, Y_binary, R, movie_data, num_recommendations=10):
    """
    Recommends movies through a 3-stage filtering process:
    1. Top 1000 by model probability
    2. Top 100 by total ratings (popularity)
    3. Top 10 by "like ratio" (likes / total ratings)

    Args:
        X: Item feature matrix (movies x features)
        W: User parameter matrix (users x features)
        b: User bias vector (1 x users)
        Y_binary: Binary rating matrix (movies x users), values: 0, 1, 0.5
        R: Rating mask matrix (1 = rated, 0 = unrated)
        movie_data: DataFrame with movie titles
        num_recommendations: Number of final recommendations

    Returns:
        List of recommended movie dictionaries with metrics
    """
    # 1. Get raw prediction probabilities
    logits = tf.matmul(X, W, transpose_b=True) + b
    probabilities = tf.sigmoid(logits).numpy()
    
    # 2. Get probabilities for new user (first column)
    user_idx = 0
    user_probabilities = probabilities[:, user_idx]

    # 3. Stage 1: Select top 1000 by model probability
    stage1_indices = np.argsort(-user_probabilities)[:1000]

    # 4. Compute popularity metrics for stage 1 movies
    results = []
    for idx in stage1_indices:
        # Count total ratings (non-unrated items)
        rated_mask = (Y_binary[idx] != 0.5)
        total_ratings = np.sum(rated_mask)
        
        # Count likes (where rating == 1)
        likes = np.sum(Y_binary[idx][rated_mask] == 1)
        
        # Skip movies with no ratings
        if total_ratings == 0:
            continue
            
        like_ratio = likes / total_ratings
        
        results.append({
            'movie_id': idx,
            'title': movie_data.iloc[idx]['title'],
            'model_prob': float(user_probabilities[idx]),
            'total_ratings': int(total_ratings),
            'like_ratio': float(like_ratio)
        })

    # 5. Stage 2: Sort by total ratings (popularity)
    stage2_results = sorted(results, key=lambda x: x['total_ratings'], reverse=True)[:100]

    # 6. Stage 3: Sort by like ratio (quality)
    stage3_results = sorted(stage2_results, key=lambda x: x['like_ratio'], reverse=True)[:num_recommendations]

    return stage3_results

# Example Usage
# Assuming you've already defined:
# - X, W, b: Model weights
# - Y_binary: Transformed ratings (0/1/0.5)
# - R: Rating mask
# - movieList_df: DataFrame with movie titles

recommendations = recommend_top_movies_with_quality(X, W, b, binaryY, R, movieList_df)

# Display Results
print("\n🏆 Top 10 High-Quality, Popular Movies You're Likely to Enjoy:\n")
for i, r in enumerate(recommendations, 1):
    print(f"{i}. {r['title']}")
    print(f"   Model Confidence: {r['model_prob']:.2f}")
    print(f"   Total Ratings: {r['total_ratings']}")
    print(f"   Like Ratio: {r['like_ratio']:.1%}")
    print("-" * 60)

'''



