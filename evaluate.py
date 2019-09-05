import numpy as np


def load_data(name):
    return np.load(f'{name}.npy')


def evaluate(query_feature, query_label, query_cam, gallery_features, gallery_labels, gallery_cams):
    cosine_distances = np.sum(query_feature * gallery_features, axis=1)
    distance_rank_index = np.flip(np.argsort(cosine_distances))

    query_index = np.argwhere(query_label == gallery_labels).flatten()
    camera_index = np.argwhere(query_cam == gallery_cams).flatten()

    # good index: index which have same label to query and diffierent camera id
    good_index = np.setdiff1d(query_index, camera_index)
    # bad label
    junk_index1 = np.argwhere(gallery_labels == -1)
    # same camera id to query
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    rank_index = distance_rank_index[np.isin(
            distance_rank_index, junk_index, assume_unique=True, invert=True)]

    ap, cmc = cal_ap_cmc(good_index, rank_index, len(gallery_labels))
    return ap, cmc


def evaluateV2(query_feature, query_label, gallery_features, gallery_labels):
    diffs = np.reshape(query_feature, (1, 1, -1)) - np.expand_dims(gallery_features, axis=0)
    distances = np.sqrt(np.sum(np.square(diffs), axis=-1) + 1e-12)
    distance_rank_index = np.argsort(distances)

    query_index = np.argwhere(query_label == gallery_labels).flatten()

    good_index = query_index

    rank_index = distance_rank_index.flatten()

    ap, cmc = cal_ap_cmc(good_index, rank_index, len(gallery_labels))
    return ap, cmc

def cal_ap_cmc(good_index, rank_index, gallery_len):
    good_rank = np.argwhere(np.isin(rank_index, good_index))
    good_rank = good_rank.flatten()
    good_len = len(good_index)
    
    ap = 0

    if good_len == 0:
        return ap, None

    cmc = np.zeros(gallery_len)
    cmc[good_rank[0]:] = 1

    for i in range(good_len):
        d_recall = 1 / good_len
        precision = (i + 1) / (good_rank[i] + 1)
        if good_rank[i] != 0:
            old_precision = i / good_rank[i]
        else:
            old_precision = 1
        ap += d_recall*(old_precision + precision)/2

    return ap, cmc

def cal_ap_v0(good_index, rank_index):
    good_rank = rank_index[np.argwhere(np.isin(rank_index, good_index))]
    good_rank = good_rank.flatten()

    ap = 0

    for i in range(len(good_index)):
        precision = (i + 1) / (good_rank[i] + 1)
        ap += precision / len(good_index)

    return ap

        

if __name__ == "__main__":
    query_features = load_data('query_features')
    query_labels = load_data('query_labels')

    gallery_features = load_data('gallery_features')
    gallery_labels = load_data('gallery_labels')

    mAP_sum = 0
    cmc_sum = np.zeros(len(gallery_labels))

    for i, (query_feature, query_label) in enumerate(zip(query_features, query_labels)):
        ap, cmc = evaluateV2(query_feature, query_label, gallery_features, gallery_labels)
        if cmc is None:
            continue
        mAP_sum += ap
        cmc_sum += cmc
        print(f'{i}/{len(query_labels)} {query_label}')
    mAP = mAP_sum / len(query_labels)
    cmc = cmc_sum / len(query_labels)
    print(f'{mAP} {cmc[0]} {cmc[2]} {cmc[4]}')
    # evaluate_classiyer()
