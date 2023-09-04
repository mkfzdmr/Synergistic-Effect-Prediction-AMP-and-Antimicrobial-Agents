import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#CLASSIFIERS
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.mixture import GaussianMixture


def get_best_models():
    models = []
    models.append(('MLPC', MLPClassifier()))
    models.append(('LGBMC',LGBMClassifier()))
    models.append(('LGBMC-optimized',LGBMClassifier(learning_rate=0.1, max_depth=20, max_features=0.7, min_samples_leaf=7, min_samples_split=5,
                            n_estimators=100, subsample=0.8) ))
    models.append(('RFC', RandomForestClassifier()))

    return models

def get_models():
    models = []

    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))

    models.append(('GPC',GaussianProcessClassifier()))

    models.append(('XGBC',XGBClassifier()))
    models.append(('LGBMC',LGBMClassifier()))

    models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('ETC', ExtraTreeClassifier()))

    models.append(('GNB', GaussianNB()))
    models.append(('BNB', BernoulliNB()))

    models.append(('SVM', SVC(gamma='auto',probability=True)))

    models.append(('BC', BaggingClassifier()))
    models.append(('ABC', AdaBoostClassifier()))
    models.append(('HGBC', HistGradientBoostingClassifier()))
    models.append(('RFC', RandomForestClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('MLPC', MLPClassifier()))

    return models
