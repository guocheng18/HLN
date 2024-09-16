import csv
import operator
import pickle
import random
import time

random.seed(42)

with open("yoochoose/raw/yoochoose-clicks.dat", "r") as f:
    reader = csv.DictReader(f, fieldnames=['SessionId', 'Time', 'ItemId', 'Category'], delimiter=",")
    sess_clicks = {}  # session_id => list of clicked items (item_id, click_time)
    sess_date = {}  # session_id => date of session
    ctr = 0
    curid = -1  # For finding new sessions
    curdate = None
    for data in reader:
        sessid = data["SessionId"]
        # Triggered when read a new session
        if curdate and not curid == sessid:
            sess_date[curid] = curdate
        curid = sessid
        curdate = time.mktime(time.strptime(data["Time"][:19], "%Y-%m-%dT%H:%M:%S"))
        item = data["ItemId"]
        if sessid in sess_clicks:
            sess_clicks[sessid] += [(item, curdate)]
        else:
            sess_clicks[sessid] = [(item, curdate)]
        ctr += 1
        if ctr % 100000 == 0:
            print("Loaded", ctr)
    sess_date[curid] = curdate

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Calculate click time interval
intervals = []
for s in list(sess_clicks):
    for idx, (item, time) in enumerate(sess_clicks[s][1:], 1):
        interval = time - sess_clicks[s][idx - 1][1]
        sess_clicks[s][idx - 1] = sess_clicks[s][idx - 1][0], interval
        intervals.append(interval)

# Normalize click time interval (z-score)
min_interv = min(intervals)
max_interv = max(intervals)
for s in list(sess_clicks):
    for idx in range(len(sess_clicks[s]) - 1):
        norm_interval = (sess_clicks[s][idx][1] - min_interv) / (max_interv - min_interv)
        sess_clicks[s][idx] = sess_clicks[s][idx][0], norm_interval

# Count number of times each item appears
iid_counts = {}  # ItemId => Occurrence times
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid, _ in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

# Delete items whose occur less than 5 times and delete sessions whose length become less than 2
for s in list(sess_clicks):
    curseq = sess_clicks[s]  # {(item_id, click_time),...]
    filseq = list(filter(lambda i: iid_counts[i[0]] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())  # Tuple list
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 1 days for test
splitdate = maxdate - 86400
print("Split date", splitdate)
train_sess = list(filter(lambda x: x[1] < splitdate, dates))  # dates: [(session_id, date),...]
test_sess = list(filter(lambda x: x[1] > splitdate, dates))

# Sort sessions by date
train_sess = sorted(train_sess, key=operator.itemgetter(1))
test_sess = sorted(test_sess, key=operator.itemgetter(1))
print(len(train_sess))
print(len(test_sess))

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
item_ctr = 1
train_seqs = []
train_dates = []
# Convert training sessions to sequences and renumber items to start from 1 !!!
for s, date in train_sess:
    seq = sess_clicks[s]
    outseq = []
    for item, interval in seq:
        if item in item_dict:
            outseq += [(item_dict[item], interval)]
        else:
            outseq += [(item_ctr, interval)]
            item_dict[item] = item_ctr
            item_ctr += 1
    if len(outseq) < 2:  # Doesn't occur
        continue
    train_seqs += [outseq]
    train_dates += [date]

test_seqs = []
test_dates = []
# Convert test sessions to sequences, ignoring items that do not appear in training set
# Note: If an item don't appear in training set, its belonging session should be removed!
for s, date in test_sess:
    seq = sess_clicks[s]
    outseq = []
    for item, interval in seq:
        if item in item_dict:
            outseq += [(item_dict[item], interval)]
    if len(outseq) < 2:
        continue
    test_seqs += [outseq]
    test_dates += [date]

print("Total items in training data: %s" % item_ctr)


# (1,2,3,4,5) => [(1,2,3,4),5]、[(1,2,3),4]、[(1,2),3]、[(1),2]
def process_seqs(iseqs, idates):
    out_seqs = []
    out_seqs_ci = []  # click intervals (seconds)
    out_dates = []
    labs = []
    for seq, date in zip(iseqs, idates):
        items = list(map(operator.itemgetter(0), seq))
        intervals = list(map(operator.itemgetter(1), seq))
        for i in range(1, len(seq)):
            labs += [items[-i]]
            out_seqs += [items[:-i]]
            out_seqs_ci += [intervals[:-i]]
            out_dates += [date]
    return out_seqs, out_seqs_ci, out_dates, labs


tr_seqs, tr_seqs_ci, tr_dates, tr_labs = process_seqs(train_seqs, train_dates)
te_seqs, te_seqs_ci, te_dates, te_labs = process_seqs(test_seqs, test_dates)

# Make train/valid/test
split_r4 = int(len(tr_seqs) / 4)
split_r64 = int(len(tr_seqs) / 64)

train_r4 = (tr_seqs[-split_r4:], tr_labs[-split_r4:])
train_r64 = (tr_seqs[-split_r64:], tr_labs[-split_r64:])
test = (te_seqs, te_labs)

# Save data
with open(f"yoochoose/processed/train_r4.pkl", "wb") as f:
    pickle.dump(train_r4, f)
with open(f"yoochoose/processed/train_r64.pkl", "wb") as f:
    pickle.dump(train_r64, f)
with open(f"yoochoose/processed/test.pkl", "wb") as f:
    pickle.dump(test, f)

print("Done.")
