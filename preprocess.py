import sys
import os
import codecs
import numpy as np
from tqdm import tqdm_notebook

def dist_remove(strokes, height=128):
    flatten = [p for s in strokes for p in s]
    flatten = np.reshape(np.array(flatten, dtype=np.float32), (-1, 2))
    min_shape = np.min(flatten, axis=0)
    flatten[:, 0] -= min_shape[0]
    flatten[:, 1] -= min_shape[1]
    max_shape = np.max(flatten, axis=0)
    ratio = height / max_shape[1]

    threshold = 1
    res = []
    for s in strokes:
        if len(s) == 0:
            continue
        stroke = [s[0], s[1]]
        for i in range(2, len(s), 2):
            delta_x = (s[i] - stroke[-2])*ratio
            delta_y = (s[i+1] - stroke[-1])*ratio
            dist = (delta_x**2 + delta_y**2)**0.5
            if dist < threshold:
                continue
            stroke.append(s[i])
            stroke.append(s[i+1])
        res.append(stroke)
    return res

def coordinate_norm(strokes):
    def line_len(x1, y1, x2, y2):
        return ((x2-x1)**2 + (y2-y1)**2)**0.5
    sum_L, sum_px, sum_py = 0, 0, 0
    for s in strokes:
        for i in range(0, len(s) - 2, 2):
            L = line_len(*s[i:i+4])
            sum_L += L
            sum_px += 0.5 * L * (s[i] + s[i+2])
            sum_py += 0.5 * L * (s[i+1] + s[i+3])
    mu_x = sum_px / sum_L
    mu_y = sum_py / sum_L

    sum_dx = 0
    for s in strokes:
        for i in range(0, len(s) - 2, 2):
            L = line_len(*s[i:i+4])
            dx = 1.0 / 3 * L * \
                 ((s[i+3]-mu_x)**2 + (s[i]-mu_x)**2 +
                  (s[i]-mu_x)*(s[i+3]-mu_x))
            sum_dx += dx
    sigma_x = (sum_dx / sum_L)**0.5
    res = []
    for s in strokes:
        stroke = []
        for i in range(0, len(s), 2):
            stroke.append((s[i] - mu_x) / sigma_x)
            stroke.append((s[i+1] - mu_y) / sigma_x)
        res.append(stroke)
    return res


def extract_online_feature(strokes, save_dir):
    fp = open(save_dir, 'w')
    for s in strokes:
        for i in range(0, len(s), 2):
            delta_x = s[i] - s[i-2] if i >= 2 else 0
            delta_y = s[i+1] - s[i-1] if i >= 2 else 0
            delta2_x = s[i] - s[i-4] if i >= 4 else 0
            delta2_y = s[i+1] - s[i-3] if i >= 4 else 0
            is_end = 1 if i + 2 == len(s) else 0
            fp.write('%.6f %.6f %.6f %.6f %.6f %.6f %d %d\n' % \
                     (s[i], s[i+1], delta_x, delta_y, delta2_x, delta2_y, is_end^1, is_end))
    fp.close()


def from_numpy_to_list(numpy_strokes):
    strokes = []
    stroke = []
    for i in range(numpy_strokes.shape[0]):
        stroke.append(numpy_strokes[i][0])
        stroke.append(numpy_strokes[i][1])
        if numpy_strokes[i][2] == 1:
            strokes.append(list(stroke))
            stroke = []
    return strokes


def read_data(data_dir):
    all_raw_feature_files = [x for x in os.listdir(data_dir) if x.endswith('.raw_feature')]
    point_seqs = []
    point_seq = []
    point = []
    tag_seqs = []
    for i, f in tqdm_notebook(enumerate(all_raw_feature_files)):
        file_name = f.split(".")[0]
        with codecs.open(os.path.join(data_dir, f), 'r', 'utf-8') as fp:
            for line in fp:
                feature = [float(x) for x in line.split()]
                point.append(feature)
                if feature[-1] == 1:
                    point_seq.append(point[:])
                    point = []
        point_seqs.append(point_seq[:])
        point_seq = []
        tag_file = os.path.join(data_dir, file_name + ".label")
        with codecs.open(tag_file, 'r', 'utf-8') as fp:
            lines = fp.readlines()
            tag_seqs.append([int(x.strip()) for x in lines])
        try:
            assert(len(point_seqs[-1]) == len(tag_seqs[-1]))
        except:
            print('file_name: %s' % file_name)
            print(len(point_seqs[-1]))
            print(len(tag_seqs[-1]))
        
    return point_seqs, tag_seqs

def generate_time_edge(list_strokes, save_dir):
    num_node = len(list_strokes)
    with open(save_dir, 'w') as fp:
        for i in range(num_node - 1):
            fp.write("%d %d\n" % (i, i + 1))
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage python preprocess.py (train/valid/test) (f/e)")
        sys.exit(0)
    home = os.environ['HOME']
    data_path = os.path.join(home, "data/IAMonDo/cls_5class")
    data_path = os.path.join(data_path, sys.argv[1])
    operation = sys.argv[2]
    
    # from points to point features
    all_point_feature = [x for x in os.listdir(data_path) if x.endswith('.point_feature')]
    for f in all_point_feature:
        file_name = f.split(".")[0]
        read_dir = os.path.join(data_path, f)
        save_dir = os.path.join(data_path, file_name + ".raw_feature")
        numpy_strokes = np.loadtxt(read_dir, dtype=np.float32)
        list_strokes = from_numpy_to_list(numpy_strokes)
        
        if 'f' in operation:
            dist_remove_strokes = dist_remove(list_strokes)
            norm_strokes = coordinate_norm(dist_remove_strokes)
            extract_online_feature(norm_strokes, save_dir)
        
        if 'e' in operation:
            time_save_dir = os.path.join(data_path, file_name + ".time_edge")
            generate_time_edge(list_strokes, time_save_dir)