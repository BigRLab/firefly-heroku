from sklearn.externals import joblib

clf = joblib.load('clf_dump.pkl')

def predict(a):
    return int(clf.predict(a)[0])

if __name__ == "__main__":
    print(predict([float(a) for a in input().split(" ")]))
