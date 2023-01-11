import json
filename = '/home/ginger/merlot_reserve/finetune/datasets/nextqa/txt_nextqa_blip_clip_loss.json'
# split = {
#     'train': 'tvqa_train.jsonl',
#     'val': 'tvqa_val.jsonl',
#     'test': 'tvqa_test_public.jsonl',
# }[split]
# data = []
# with open(split_fn, 'r') as f:
#     for idx, l in enumerate(f):
#         if idx % num_folds != fold:
#             continue
#         item = json.loads(l)
#         item['ts'] = tuple([float(x) for x in item['ts'].split('-')])
#         assert len(item['ts']) == 2
#         if np.any(np.isnan(item['ts'])):
#             item['ts'] = (0, 9999.0)
#         data.append(item)

# data = []
# with open(filename, 'r') as f:
#     for idx, l in enumerate(f):
#         item = json.loads(l)['train']
#         data.append(item)
#         __import__('ipdb').set_trace()

#         for i in range(5):
#             sx = f'option_{str(i)}'
#             x = item.pop(sx) 
import os
num_folds = 8
data_dir = '/home/ginger/merlot_reserve/finetune/datasets/nextqa/subtitles/txt_nextqa_blip_clip_loss.json'

data=[]
with open(data_dir, 'r') as f:
    for idx, l in enumerate(f):
        data = json.loads(l)['train']

data = [val for val in data for _ in (0, 1)]

i = 0
import time
for fold in range(num_folds):
    start = time.time()
    out_fn = os.path.join('/home/ginger/merlot_reserve/finetune/datasets/nextqa/', '{}{:03d}of{:03d}.tfrecord'.format('train', fold, num_folds))
    with open(out_fn, 'w') as f:

        while True:
            print(out_fn)
            f.write(str(data[i]))
            i += 1
            if i % (len(data)//num_folds) == 0:
                break
    end = time.time()
    print(end-start)
    