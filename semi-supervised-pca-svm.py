#!/usr/bin/env python
"""
Documentation: http://scikit-learn.org/stable/modules/svm.html

Notes:

Support Vector Machines are supervised learning methods for classification, regression, and outliers detection.

Advantages:

SVMs are effective in high-dimensional spaces, even when the number of dimensions is greater than the number of samples.

SVMs are memory-efficient because they only use a subset of training examples (support vectors) in the decision function.

SVMs use a kernel for the decision function.

Disadvantages:

SVMs do not directly provide probability estimates for predictions, but these can be obtained through cross validation.
However, this degrades performance.

The number of features should not greatly exceed the number of examples, or performance will be poor.

Implementation:

This approach uses Rafael's (https://www.kaggle.com/c/data-science-london-scikit-learn/forums/t/4986/ideas-that-worked-or-did-not)
suggestion to use semi-supervised learning, and Martin Mevald's
(https://www.kaggle.com/c/data-science-london-scikit-learn/visualization/1239) example that uses PCA.

The data is curiously split into 1 parts training for every 9 parts test. Martin's
Labels are predicted for the test set. The test set is then stacked on to the training data, and the predictions are
appended to the training labels.

"""
import numpy as np
from sklearn import cross_validation as cv
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy import sparse


def write_submission(result):
    f = open('result2.csv', 'w')
    f.write('Id,Solution\n')
    count = 1
    for x in result:
        f.write('%d,%d\n' % (count, x))
        count += 1
    f.close()


def predict(train, target, test):
    pca = PCA(n_components=12, whiten=True)
    test = pca.fit_transform(test)
    train = pca.transform(train)

    clf = SVC(
        C=1000000, cache_size=200, class_weight=None, coef0=0.0, degree=3,
        gamma=0.277777777778, kernel='rbf', max_iter=-1, probability=False,
        random_state=None, shrinking=True, tol=0.001, verbose=False
    )

    clf.fit(train, target)

    # Estimate score
    scores = cv.cross_val_score(clf, train, target, cv=60)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    results = clf.predict(test)
    return results


def predict2(train, target, test):
    clf = SVC(
        C=1000000, cache_size=200, class_weight=None, coef0=0.0, degree=3,
        gamma=0.277777777778, kernel='rbf', max_iter=-1, probability=False,
        random_state=None, shrinking=True, tol=0.001, verbose=False
    )

    clf.fit(train, target)
    scores = cv.cross_val_score(clf, train, target, cv=60)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    results = clf.predict(test)
    return results


def main():
    loadData = lambda f: np.genfromtxt(open(f, 'r'), delimiter=',')
    train = loadData('train.csv')
    target = loadData('trainLabels.csv')
    test = loadData('test.csv')
    results = predict(train, target, test)
    train2 = sparse.vstack([train, test])
    print train.shape, train2.shape
    target2 = np.append(target, results)
    results2 = predict2(train2.toarray(), target2, test)
    write_submission(results2)

if __name__ == '__main__':
    main()