import pandas as pd

def personalizedMethdo(people, model_name):
    for person in people:
        true_data = pd.read_csv(person + "/Data/True/All.csv")
        false_data = pd.read_csv(person + "Data/False/All.csv")

        X_train, X_test, y_train, y_test = separate_data(true_data, false_data)
        model = get_model(model_name)

        model.fit(X_train, y_train)
        y_predict = model.predict(X_test, y_test)

        y_predict - 
