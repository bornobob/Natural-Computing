from sklearn.metrics import auc
from subprocess import run, PIPE
import matplotlib.pyplot as plt
import os.path

NEG_SEL_DIR = 'negative-selection'
JAR_PATH = os.path.join(NEG_SEL_DIR, 'negsel2.jar')
ENGLISH_TRAIN = os.path.join(NEG_SEL_DIR, 'english.train')
ENGLISH_TEST = os.path.join(NEG_SEL_DIR, 'english.test')
TAGALOG_TEST = os.path.join(NEG_SEL_DIR, 'tagalog.test')
HILIGAYNON_TEST = os.path.join(NEG_SEL_DIR, 'lang', 'hiligaynon.txt')
MIDDLE_ENGLISH_TEST = os.path.join(NEG_SEL_DIR, 'lang', 'middle-english.txt')
PLAUTDIETSCH_TEST = os.path.join(NEG_SEL_DIR, 'lang', 'plautdietsch.txt')
XHOSA_TEST = os.path.join(NEG_SEL_DIR, 'lang', 'xhosa.txt')

with open(ENGLISH_TEST, 'r') as f:
    english_test = f.read()
with open(TAGALOG_TEST, 'r') as f:
    tagalog_test = f.read()
with open(HILIGAYNON_TEST, 'r') as f:
    hiligaynon_test = f.read()
with open(MIDDLE_ENGLISH_TEST, 'r') as f:
    middle_english_test = f.read()
with open(PLAUTDIETSCH_TEST, 'r') as f:
    plautdietsch_test = f.read()
with open(XHOSA_TEST, 'r') as f:
    xhosa_test = f.read()


def anomaly_scores(n, r, anomalous_lang):
    p = run(['java', '-jar', JAR_PATH, '-self', ENGLISH_TRAIN, '-n', str(n),
             '-r', str(r), '-c', '-l'],
            stdout=PIPE, input=english_test, encoding='ascii')
    results_eng = list(map(lambda x: float(x.strip('\n ')),
                           filter(lambda x: x.strip('\n ') != '',
                                  p.stdout.split('\n'))))

    t = run(['java', '-jar', JAR_PATH, '-self', ENGLISH_TRAIN, '-n', str(n),
             '-r', str(r), '-c', '-l'],
            stdout=PIPE, input=anomalous_lang, encoding='ascii')
    results_tag = list(map(lambda x: float(x.strip('\n ')),
                           filter(lambda x: x.strip('\n ') != '',
                                  t.stdout.split('\n'))))

    results = [(x, 0) for x in results_eng] + [(x, 1) for x in results_tag]
    unique_scores = set(x[0] for x in results)
    return results, unique_scores


def calculate_auc(results, unique_scores, show_plot=False):
    tprs = []
    fprs = []

    for s in sorted(list(unique_scores)):
        tp = len(list(filter(lambda x: x[0] < s and x[1] == 0, results)))
        fp = len(list(filter(lambda x: x[0] < s and x[1] == 1, results)))

        tn = len(list(filter(lambda x: x[0] >= s and x[1] == 1, results)))
        fn = len(list(filter(lambda x: x[0] >= s and x[1] == 0, results)))

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tprs.append(tpr)
        fprs.append(fpr)

    if show_plot:
        plt.plot(fprs, tprs)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.plot([0, 1], [0, 1], '--')
        plt.show()

    return auc(fprs, tprs)


if __name__ == '__main__':
    lang_dict = {'Tagalog': tagalog_test,
                 'Hiligaynon': hiligaynon_test,
                 'Middle English': middle_english_test,
                 'Plautdietsch': plautdietsch_test,
                 'Xhosa': xhosa_test}
    for lang, vals in lang_dict.items():
        print('Executing language', lang)
        for i in range(1, 10):
            results, unique_results = anomaly_scores(10, i, vals)
            auc_result = calculate_auc(results, unique_results)
            print('r =', i, 'score =', auc_result)
