import pandas as pd

# Create dataset and save to CSV
def create_csv():
    data = {
        "Weather": ["Sunny", "Sunny", "Rainy", "Sunny"],
        "Temperature": ["Warm", "Warm", "Cold", "Warm"],
        "Humidity": ["Normal", "High", "High", "High"],
        "Wind": ["Strong", "Strong", "Strong", "Weak"],
        "PlayTennis": ["Yes", "Yes", "No", "Yes"]
    }

    df = pd.DataFrame(data)
    df.to_csv("training_data.csv", index=False)
    return df

# Find-S algorithm
def find_s(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    hypothesis = None

    # First positive example
    for i in range(len(y)):
        if y[i] == "Yes":
            hypothesis = X[i].copy()
            break

    # Generalize using positive examples
    for i in range(len(y)):
        if y[i] == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != X[i][j]:
                    hypothesis[j] = '?'

    return hypothesis

# Main
df = create_csv()

print("Training Data:")
print(df)

h = find_s(df)

print("\nFinal Hypothesis:")
print(h)
