from sklearn.metrics import auc
from subprocess import run, PIPE
import matplotlib.pyplot as plt
import os.path

NEG_SEL_DIR = 'negative-selection'
SYSCALLS_DIR = os.path.join(NEG_SEL_DIR, 'syscalls')
SND_CERT_DIR = os.path.join(SYSCALLS_DIR, 'snd-cert')
SND_UNM_DIR = os.path.join(SYSCALLS_DIR, 'snd-unm')
JAR_PATH = os.path.join(NEG_SEL_DIR, 'negsel2.jar')
CERT_TRAIN = os.path.join(SND_CERT_DIR, 'snd-cert.train')
CERT_CHUNK_SIZE = 7
CERT_TEST_SETS = 3
CERT_TEST_NAME = 'snd-cert.{}.test'
CERT_LABEL_NAME = 'snd-cert.{}.labels'
CERT_TESTS = [os.path.join(SND_CERT_DIR, CERT_TEST_NAME.format(i))
              for i in range(1, CERT_TEST_SETS + 1)]
CERT_LABELS = [os.path.join(SND_CERT_DIR, CERT_LABEL_NAME.format(i))
               for i in range(1, CERT_TEST_SETS + 1)]
UNM_TRAIN = os.path.join(SND_UNM_DIR, 'snd-unm.train')
UNM_CHUNK_SIZE = 7
UNM_TEST_SETS = 3
UNM_TEST_NAME = 'snd-unm.{}.test'
UNM_LABEL_NAME = 'snd-unm.{}.labels'
UNM_TESTS = [os.path.join(SND_UNM_DIR, UNM_TEST_NAME.format(i))
             for i in range(1, UNM_TEST_SETS + 1)]
UNM_LABELS = [os.path.join(SND_UNM_DIR, UNM_LABEL_NAME.format(i))
              for i in range(1, UNM_TEST_SETS + 1)]
TEMP_CERT_TRAIN = os.path.join(SND_CERT_DIR, 'snd-cert.train.tmp')
TEMP_UNM_TRAIN = os.path.join(SND_UNM_DIR, 'snd-unm.train.tmp')


def read_data(path):
    datalines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n ')
            if line:
                datalines.append(line)
    return datalines


def gen_substrings(string, length):
    return [string[i:i+length] for i in range(len(string) - length + 1)]


def split_string(string, length):
    return [string[i * length:(i + 1) * length]
            for i in range(len(string) // length)]


def preprocess_train(path, chunk_size, use_substrings=False):
    datalines = read_data(path)
    split_data = []
    for line in datalines:
        if use_substrings:
            split_data += gen_substrings(line, chunk_size)
        else:
            split_data += split_string(line, chunk_size)
    return list(set(split_data))


def preprocess_test(data_path, labels_path, chunk_size):
    datalines = read_data(data_path)
    labels = list(map(lambda x: int(x), read_data(labels_path)))
    split_data = []
    for data, label in zip(datalines, labels):
        split_data.append((split_string(data, chunk_size), label))
    return split_data


def combine_results(data, results):
    result = []
    skip = 0
    for d, label in data:
        relevant_results = results[skip:skip + len(d)]
        average = sum(relevant_results) / len(d)
        result.append((average, label))
        skip += len(d)
    return result


def anomaly_scores(n, r, train_path, data):
    input_data = '\n'.join('\n'.join(x[0]) for x in data)
    t = run(['java', '-jar', JAR_PATH, '-self', train_path, '-n', str(n),
             '-r', str(r), '-c', '-l'],
            stdout=PIPE, input=input_data, encoding='ascii')
    results = list(map(lambda x: float(x.strip('\n ')),
                       filter(lambda x: x.strip('\n ') != '',
                              t.stdout.split('\n'))))
    return combine_results(data, results)


def calculate_auc(results, show_plot=False):
    tprs = []
    fprs = []
    unique_results = set(x[0] for x in results)
    for s in sorted(list(unique_results)):
        tp = len(list(filter(lambda x: x[0] < s and x[1] == 0, results)))
        fp = len(list(filter(lambda x: x[0] < s and x[1] == 1, results)))
        tn = len(list(filter(lambda x: x[0] >= s and x[1] == 1, results)))
        fn = len(list(filter(lambda x: x[0] >= s and x[1] == 0, results)))
        tprs.append(tp / (tp + fn))
        fprs.append(fp / (fp + tn))

    if show_plot:
        plt.plot(fprs, tprs)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.plot([0, 1], [0, 1], '--')
        plt.show()

    return auc(fprs, tprs)


def preprocess_train_files():
    cert = preprocess_train(CERT_TRAIN, CERT_CHUNK_SIZE)
    unm = preprocess_train(UNM_TRAIN, UNM_CHUNK_SIZE)
    with open(TEMP_CERT_TRAIN, 'w') as f:
        f.write('\n'.join(cert))
    with open(TEMP_UNM_TRAIN, 'w') as f:
        f.write('\n'.join(unm))


if __name__ == '__main__':
    preprocess_train_files()
    print('CERT')
    for r in range(6, 8):
        for t in range(3):
            tests = preprocess_test(CERT_TESTS[t], CERT_LABELS[t], CERT_CHUNK_SIZE)
            results = anomaly_scores(CERT_CHUNK_SIZE, r, TEMP_CERT_TRAIN, tests)
            auc_result = calculate_auc(results)
            print('testset', t + 1, 'r=', r, 'score =', auc_result)
    print('UNM')
    for r in range(6, 8):
        for t in range(3):
            tests = preprocess_test(UNM_TESTS[t], UNM_LABELS[t], UNM_CHUNK_SIZE)
            results = anomaly_scores(UNM_CHUNK_SIZE, r, TEMP_UNM_TRAIN, tests)
            auc_result = calculate_auc(results)
            print('testset', t + 1, 'r=', r, 'score =', auc_result)
