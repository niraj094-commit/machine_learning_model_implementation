import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import time

# Importing the dataset_csv file

data = pd.read_csv("spam_dataset_realistic_10000.csv",encoding="latin-1")
data = data[["Email Text", "Label"]]   # Keeping only the relevant columns , although I currently have two columns!! but for learning itna to chlta hai!

# print(data.head())  # Displaying the first few rows of the dataset
# Not jruri now


## Note : Machine Learning models work with numerical data, so we need to convert the text data into numbers.
##      : By using Techques Like Bag of words, TF-IDF, etc.



print(data["Label"].value_counts())  # Displaying the count of each label (spam/not spam)

# Visualizing the distribution of Spam vs Ham messages

plt.figure(figsize=(8, 5))
ax = data['Label'].value_counts().plot(
    kind='bar',
    color=['#4CAF50', '#F44336'],  # Green for Ham, Red for Spam
    edgecolor='black',
    width=0.7
)

# Add counts on top of bars
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():,}', 
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', 
        va='center', 
        xytext=(0, 5), 
        textcoords='offset points'
    )

# Customize the plot
plt.title('Distribution of Spam vs Not_Spam Messages', fontsize=14, pad=20)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(
    ticks=[0, 1], 
    labels=['Not_Spam', 'Spam'], 
    rotation=0
)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Remove top and right spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()




# Converting Labels to numerical values ("Not spam" --> 0 and "spam" --> 1 )
data["Label"] = data["Label"].map({"not_spam": 0, "spam": 1})  


print(data["Label"].isnull().sum())  # Should be 0

data = data.dropna()    # Dropping rows with missing or Null values

## prepare data for training the model
# splitting data
x = data["Email Text"]
y = data["Label"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42,stratify=y)       # Testing with 20% data and 80% for training the model

# Converting texts to numbers
vectorizer = TfidfVectorizer(stop_words="english")
x_train_vec = vectorizer.fit_transform(x_train) # Fitting the vectorizer on the training data
x_test_vec = vectorizer.transform(x_test)   # Transforming the test data using the fitted vectorizer


# Training the model
model = MultinomialNB()
model.fit(x_train_vec,y_train)

# Making predictions
predictions = model.predict(x_test_vec)

# Checking accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy*100:.2f}%")

# Evaluating the model

print(confusion_matrix(y_test,predictions)) # Confusion matrix
print(classification_report(y_test,predictions)) # Classification report


# Saving & Testing the model
joblib.dump(model, "spam_classifier_model.pkl")  # Saving the model
joblib.dump(vectorizer, "vectorizer.pkl")  # Saving the vectorizer

loaded_model = joblib.load("spam_classifier_model.pkl")  # Loading the model
loaded_vectorizer = joblib.load("vectorizer.pkl")  # Loading the vectorizer

def detect_spam(email_text):
    email_text_vec = loaded_vectorizer.transform([email_text])
    prediction = loaded_model.predict(email_text_vec)
    return "Spam!" if prediction[0] == 1 else "Not Spam!"

time.sleep(2)

## Example usage
# print(detect_spam("Congratulations! You've won a lottery! Claim your prize now."))
# print(detect_spam("Meeting at 10 AM tomorrow."))