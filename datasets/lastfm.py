import csv
import operator
import pickle
import random
import time

random.seed(42)

# Header[SessionId;TimeStamp;ItemId]
with open("lastfm/raw/lastfm_info.csv", "r") as f:
    reader = csv.DictReader(f, delimiter=",")
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
        curdate = time.mktime(time.strptime(data["TimeStamp"], "%Y-%m-%dT%H:%M:%SZ"))
        item = data["ItemId"]
        if sessid in sess_clicks:
            sess_clicks[sessid] += [(item, curdate)]
        else:
            sess_clicks[sessid] = [(item, curdate)]
        ctr += 1
        if ctr % 100000 == 0:
            print("Loaded", ctr)
    sess_date[curid] = curdate

# collapse repeating items
new_sess_clicks = {}
for s in sess_clicks.keys():
    new_sess_clicks[s] = [sess_clicks[s][0]]
    for i in range(1, len(sess_clicks[s])):
        last_event = new_sess_clicks[s][-1]
        current_event = sess_clicks[s][i]
        if current_event != last_event:
            new_sess_clicks[s].append(current_event)
sess_clicks = new_sess_clicks

# Filter out length 1 and > 50 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1 or len(sess_clicks[s]) > 50:
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
        norm_interval = (sess_clicks[s][idx][1] - min_interv) / (
            max_interv - min_interv
        )
        sess_clicks[s][idx] = sess_clicks[s][idx][0], norm_interval

# Split out test set based on dates
dates = list(sess_date.items())  # Tuple list
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 60 days for test
splitdate = maxdate - 60 * 86400
print("Split date", splitdate)
train_sess = list(
    filter(lambda x: x[1] < splitdate, dates)
)  # dates: [(session_id, date),...]
test_sess = list(filter(lambda x: x[1] > splitdate, dates))

# Sort sessions by date
train_sess = sorted(train_sess, key=operator.itemgetter(1))
test_sess = sorted(test_sess, key=operator.itemgetter(1))
print(len(train_sess))
print(len(test_sess))

# Convert training sessions to sequences and renumber items to start from 1 !!!
item_dict = {}
item_ctr = 1
train_seqs = []
train_dates = []
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

train = (tr_seqs, tr_seqs_ci, tr_labs)
test = (te_seqs, te_seqs_ci, te_labs)

# Save data
with open(f"lastfm/processed/train.pkl", "wb") as f:
    pickle.dump(train, f)
with open(f"lastfm/processed/test.pkl", "wb") as f:
    pickle.dump(test, f)

print("Done.")
