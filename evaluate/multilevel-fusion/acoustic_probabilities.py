import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append('/mnt/afs/250010218/multi-level-Chinese-decoding/evaluate')
from embeddings_alignment.semantic_eval import evaluate_semantic_mapping
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Label list (61 words)
label_list = ['丝瓜', '你', '关门', '凳子', '厕所', '口渴', '吃',
               '喝', '嘴巴', '外卖', '头疼', '家人', '小刀', '帮助',
                  '平静', '心情', '怎样', '感觉', '愿意', '我', '手机',
                    '找', '把', '护士', '拿', '换药', '放在', '是', '有',
                          '朋友', '橙汁', '毛巾', '汤圆', '漂亮', '热水',
                            '猪肉', '玩', '电脑', '看', '碗', '穿', '篮球',
                                  '米饭', '给', '脸盆', '菠萝', '葱花', '蒜泥',
                                      '衣服', '豆腐', '软糖', '醋', '钢琴', '问题',
                                          '需要', '青菜', '面条', '音乐', '预约', '香肠', '鱼块']

def load_test_embeddings(path):
    """Load test embeddings from .npy file"""
    data = np.load(path, allow_pickle=True)
    labels = data[:, -1]
    # Convert label elements to int
    labels = labels.astype(int)
    labels = [label_list[i] for i in labels]
    labels = np.array(labels)
    embeddings = data[:, 0:-1]
    return labels, embeddings


semantic_path='evaluate/multilevel-fusion/sub01/semantic_embeddings_epoch_300.npy'
labels,_=load_test_embeddings(semantic_path)

GT_path = '/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Acoustic_label.npz'
tone1_path = 'evaluate/multilevel-fusion/sub01/tone1_test.npz'
tone2_path = 'evaluate/multilevel-fusion/sub01/tone2_test.npz'


# read npz
def load_npz_acoustic_GT(path):
    data = np.load(path, allow_pickle=True)
    return data["chars"], data["embeddings"]

def load_npz_acoustic_test(path):
    #['logits', 'labels', 'probabilities', 'predictions', 'true_labels', 'top1_accuracy', 'top2_accuracy', 'top3_accuracy', 'best_epoch', 'tone_name']
    data = np.load(path, allow_pickle=True)
    label = data['labels']
    probabilities = data['probabilities']
    return label, probabilities

chars_GT, emb_GT = load_npz_acoustic_GT(GT_path)
labels_tone1, prob_tone1 = load_npz_acoustic_test(tone1_path)

