import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
def evaluate_cluster(embeds, y, n_labels, kmeans_random_state):
    Y_pred = KMeans(n_labels, random_state=kmeans_random_state).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    return nmi, ari


def evaluate(embeds, idx_train, idx_val, idx_test, label, nb_classes, device, lr, wd, dataset, ratio, epoch, isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    #train_lbls = torch.argmax(label[idx_train], dim=-1)
    #val_lbls = torch.argmax(label[idx_val], dim=-1)
    #test_lbls = torch.argmax(label[idx_test], dim=-1)
    train_lbls = label[idx_train]
    val_lbls = label[idx_val]
    test_lbls = label[idx_test]
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []
    run_similarity_search(np.array(test_embs.cpu()), np.array(test_lbls.cpu()))

    for _ in range(10):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score = roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                  y_score=best_proba.detach().cpu().numpy(),
                                  multi_class='ovr'
                                  )
        auc_score_list.append(auc_score)


    if isTest:
        print("\t[Classification] Macro-F1: [{:.4f}, {:.4f}]  Micro-F1: [{:.4f}, {:.4f}]  auc: [{:.4f}, {:.4f}]"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
        #return np.mean(macro_f1s), np.mean(micro_f1s), np.mean(auc_score_list)
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)
    f = open("result_"+dataset+str(ratio)+".txt", "a")
    #f.write(str(starttime.strftime('%Y-%m-%d %H:%M'))+"\t")
    f.write("Epoch: " + str(epoch) + "\t")
    f.write("Ma-F1_mean: "+str(np.around(np.mean(macro_f1s)*100, 2))+" +/- "+ str(np.around(np.std(macro_f1s)*100,2))+"\t")
    f.write(" Mi-F1_mean: "+str(np.around(np.mean(micro_f1s)*100,2))+" +/- "+ str(np.around(np.std(micro_f1s)*100,2))+"\t")
    f.write(" AUC_mean: "+str(np.around(np.mean(auc_score_list)*100,2))+ " +/- "+ str(np.around(np.std(auc_score_list)*100,2))+"\n")
    f.close()
    return np.mean(macro_f1s), np.mean(micro_f1s), np.mean(auc_score_list)

def run_kmeans(x, y, k, starttime, dataset, epoch):
    estimator = KMeans(n_clusters=k,n_init="auto",random_state=0)

    NMI_list = []
    ARI_list = []
    for _ in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)

        n1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        a1 = adjusted_rand_score(y, y_pred)
        NMI_list.append(n1)
        ARI_list.append(a1)

    nmi = sum(NMI_list) / len(NMI_list)
    ari = sum(ARI_list) / len(ARI_list)

    print('\t[Clustering] Epoch: {:d} NMI: {:.2f}   ARI: {:.2f}'.format(epoch, np.round(nmi*100,2), np.round(ari*100,2)))
    f = open("result_" + dataset + "_NMI&ARI.txt", "a")
    f.write(str(starttime.strftime('%Y-%m-%d %H:%M'))+"\t Epoch: " + str(epoch) +"\t NMI: " + str(np.round(nmi*100,4)) +\
         "\t ARI: " + str(np.round(ari*100,4)) + "\n")
    f.close()
    return nmi*100, ari*100
from sklearn.metrics import pairwise

def run_similarity_search(test_embs, test_lbls):

    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10]: #
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))

    st = ','.join(st)
    print("\t[Similarity] [5,10] : [{}]".format(st))
    st = st.split(', ')
    return st