import re
import os
import random


def remove_punctuation(s: str):
    """
    remove the punctuation marks from the string
    note that ! and ' and $ will not be replaced
    """

    return s.replace("(", " ").replace(")", " ").\
        replace("[", " ").replace("]", " "). \
        replace("{", " ").replace("}", " "). \
        replace("<", " ").replace(">", " "). \
        replace(":", " ").replace(";", " "). \
        replace(",", " ").replace(".", " "). \
        replace("/", " ").replace("?", " "). \
        replace('"', " ").replace("|", " "). \
        replace("`", " ").replace("~", " "). \
        replace("@", " ").replace("#", " "). \
        replace("%", " ").replace("!", " !"). \
        replace("^", " ").replace("&", " "). \
        replace("*", " ").replace("-", " "). \
        replace("_", " ").replace("+", " "). \
        replace("=", " ").replace("\\", " ")


def remove_empty_n_number(s: list[str]):
    """
    remove all empty strings
    replace all numbers with /NUM1, /NUM2, /NUM3, /NUM3+
    """

    result = []
    for item in s:
        if re.fullmatch(r'\d+', item):  # Check if item is a pure number
            length: int = len(item)
            if length == 1:
                result.append(r"/NUM1")
            elif length == 2:
                result.append(r"/NUM2")
            elif length == 3:
                result.append(r"/NUM3")
            else:
                result.append(r"/NUM3+")
        elif item:  # Remove empty strings
            result.append(item.lower())
    return result


def remove_empty_n_lower(s: list[str]):
    """
    remove all empty strings
    """

    result = [item.lower() for item in s if item]
    return result


def extract_content(mail: str) -> list[str]:
    """
    get the list with only the mail content
    """

    context: str = "\n".join(mail.split('\n\n')[1:])
    mail_text = remove_punctuation(context)
    words: list[str] = remove_empty_n_lower(re.split(r"\s+|\n", mail_text))
    return words


def extract_header_n_content(mail: str) -> list[str]:
    """
    get the list with the mail content ("From" and "Subject" in header, and the mail body)
    """

    meta: str = mail.split('\n\n')[0]
    # grab the the starts with "From" and "Subject"
    meta: str = "\n".join(["".join(line.split(": ")[1:])
                           for line in meta.split('\n') if line.startswith("From") or line.startswith("Subject")])
    context: str = "\n".join(mail.split('\n\n')[1:])
    mail_text: str = remove_punctuation(meta) + remove_punctuation(context)
    # separate the words with space or \n and place them into a list
    words: list[str] = remove_empty_n_number(re.split(r"\s+|\n", mail_text))
    return words


def build_cross_validate(dst_path: str, k: int, total_data: int, header: bool = False):
    """
    build the cross validation set using "./data/*" and k,
    save to "./cross_validation/"
    """

    idx_path: str = "./label/index"
    labels: list[str] = open(idx_path, 'r').readlines()

    # Step 1: Collect all file paths
    file_paths = []
    for j in range(126):
        src_path = f"./data/{j:03}/"  # Cleaner formatting
        for f in sorted(os.listdir(src_path)):
            file_paths.append(f'{src_path}{f}')  # Store (folder, filename) pairs

    # Step 2: Shuffle files randomly
    random.shuffle(file_paths)

    data_per_set: int = total_data // k
    count = 0
    cur_cv_file = 0

    # Step 3: Write to files
    dst_f = open(f'{dst_path}cv{cur_cv_file}', "w+")
    for src_path in file_paths:
        mail = open(f"{src_path}", encoding='utf-8', errors='replace').read()
        mail_text = extract_header_n_content(mail) if header else extract_content(mail)
        match_line = [line for line in labels if line.endswith(f'{src_path}\n')]
        label = 1 if match_line[0].split(" ")[0] == "spam" else 0
        string_mail = ",".join(mail_text)
        dst_f.write(f"{label}: {string_mail}\n")
        count += 1

        if data_per_set == count:
            count = 0
            cur_cv_file += 1
            dst_f.close()
            if cur_cv_file == k:
                return
            dst_f = open(f'{dst_path}cv{cur_cv_file}', "w+")


def get_mails_labels(path: str):
    """
    get the mails and labels from the file
    """

    file = open(path, 'r').readlines()
    labels = [int(line.split(": ")[0]) for line in file]
    mails = [line.split(": ")[1].split(",") for line in file]
    return mails, labels


def compute_metrics(tp, fp, fn, total):
    """
    Compute precision, recall, F1-score, and accuracy
    """

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp / total if total > 0 else 0

    return precision, recall, f1_score, accuracy


def k_cross_validation(path: str, k: int, classifier):
    """
    Perform k-fold cross-validation
    """

    acc_sum, tp_sum, fp_sum, fn_sum, total_samples = 0, 0, 0, 0, 0

    for i in range(k):
        # Train the classifier
        for j in [_ for _ in range(k) if _ != i]:
            cv_path = f"{path}cv{j}"
            mails, labels = get_mails_labels(cv_path)
            classifier.train(mails, labels)

        # Validate the classifier
        val_path = f"{path}cv{i}"
        mails, labels = get_mails_labels(val_path)
        correct, tp, fp, fn = 0, 0, 0, 0
        for j in range(len(mails)):
            pred = classifier.predict(mails[j])
            actual = labels[j]
            if pred == actual:
                correct += 1
                tp += 1
            else:
                fp += 1 if pred == 1 else 0
                fn += 1 if actual == 1 else 0

        accuracy = correct / len(mails)
        print(f"Fold {i} - Accuracy: {accuracy:.4f}")
        print(f"Fold {i} - Precision: {tp / (tp + fp):.4f}, Recall: {tp / (tp + fn):.4f}, F1-score: {(2 * tp / (2 * tp + fp + fn)):.4f}")
        acc_sum += accuracy
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        total_samples += len(mails)
        classifier.clean()

    # Compute overall metrics
    precision, recall, f1_score, avg_accuracy = compute_metrics(tp_sum, fp_sum, fn_sum, total_samples)
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")


def train_and_validate(classifier, cross_validation_path, train_idx, val_idx_list):
    """
    Helper function to train and validate the classifier
    """

    train_mails, train_labels = get_mails_labels(f"{cross_validation_path}cv{train_idx}")
    classifier.train(train_mails, train_labels)

    tp, fp, fn, total = 0, 0, 0, 0
    for i in val_idx_list:
        val_mails, val_labels = get_mails_labels(f"{cross_validation_path}cv{i}")
        for j in range(len(val_mails)):
            pred = classifier.predict(val_mails[j])
            actual = val_labels[j]
            if pred == actual:
                tp += 1
            else:
                fp += 1 if pred == 1 else 0
                fn += 1 if actual == 1 else 0
            total += 1

    precision, recall, f1_score, accuracy = compute_metrics(tp, fp, fn, total)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
    classifier.clean()


if __name__ == '__main__':
    seed: int = 100
    random.seed(seed)

    cross_validation_path: str = "./cross_validation/"
    total_data: int = 37823
    k: int = 20
    build_cross_validate(cross_validation_path, k, total_data)
