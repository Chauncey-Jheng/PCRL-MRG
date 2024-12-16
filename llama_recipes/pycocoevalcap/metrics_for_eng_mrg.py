from .bleu.bleu import Bleu
from .meteor import Meteor
from .cider.cider import Cider
from .rouge import Rouge

# import jieba
# import os
# all_medical_nounts_file = os.path.join(os.path.dirname(__file__), "all_medical_nouns.txt")
# jieba.load_userdict(all_medical_nounts_file)

# def add_space_to_each_word_Jieba(report):
#     # Using jieba to cut the Chinese word
#     word_list = list(jieba.cut(report, cut_all=False, HMM=True))
#     # Adding space between the Chinese word
#     final_report = ""
#     for w in word_list:
#         final_report = final_report + w + " "
#     return final_report

def compute_scores(gts, res, temp=False):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    chosen_score = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"])
    ]

    # replace('\n', ' ').replace('\r', ' ') to prevent meteor's errors.
    gts = {id:[caption[0].replace('\n', ' ').replace('\r', ' ')] for id, caption in gts.items()}
    res = {id:[caption[0].replace('\n', ' ').replace('\r', ' ')] for id, caption in res.items()}

    if temp is False:
        eval_res = {}
        # Compute score for each metric
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res

    if temp is True:
        eval_res = {}
        # Compute score for each metric
        for scorer, method in chosen_score:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res
