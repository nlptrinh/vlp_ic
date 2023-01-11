python nextqa_finetune.py ../../pretrain/configs/base.yaml ../model_checkpoints/base_resadapt -lr=5e-6 -ne=1 -wd=1e-7

# {'data': {'train_fns': '/home/ginger/merlot_reserve/finetune/tvqa/train000of001.tfrecord', 'num_train_files': 1, 'use_audio_token_prob': 0.5, 'random_scale_max': 1.1, 'random_scale_min': 1.0, 'fft_hop_length': 588, 'fft_window_size': 1536, 'num_mels': 64, 'sample_rate': 22050, 'spec_size': 188, 'mask_rate': 0.25, 'num_audio2text_seqs': 1, 'num_text2audio_seqs': 1, 'num_text_seqs': 1, 'num_text_seqs_in_record': 1, 'num_segments': 7, 'num_segment_groups': 2, 'num_audio_subsegments': 3, 'seq_len': 640, 'lang_seq_len': 256, 'num_text_spans_to_include': 48, 'text_span_budget': 38, 'num_answers': 5}, 'model': {'hidden_size': 768, 'joint_num_layers': 12, 'use_bfloat16': True, 'audio_num_layers': 12, 'audio_patch_size': 2, 'audio_seq_length': 60, 'audio_token_length': 6, 'output_grid': [12, 20], 'vit_patch_size': 16, 'vit_pooling_ratio': 2, 'vit_num_layers': 12, 'span_num_layers': 4, 'text_span_length': 15}, 'device': {'use_tpu': True, 'num_tpu_cores': 512, 'output_dir': '/home/ginger/merlot_reserve/finetune/tvqa/tvqa_outputs/base.yaml/2023-01-09-00:30.02', 'batch_size': 32, 'iterations_per_loop': 2, 'commit_every_nsteps': 50, 'n_fns_per_cycle': 1, 'num_parallel_reads': 128, 'shuffle_buffer_size': 4096, 'wandb_project': 'merlotreserve', 'prefetch_size': 0}, 'optimizer': {'beta_2': 0.98, 'eps': 1e-06, 'learning_rate': 5e-06, 'num_train_steps': 2, 'num_warmup_steps': 1, 'use_bfloat16_adam': True, 'weight_decay_rate': 1e-07, 'do_bias_correction': True}, '_ckpt': '/home/ginger/merlot_reserve/model_checkpoints/base_resadapt'}


# # TVQA
# {"a0": "A dog with a broken leg.", "a1": "Stolen electronics.", "a2": "Bunnies.", "a3": "A romantic dinner.", "a4": "A desk to put together.", "answer_idx": 2, "q": "What was in the back of Zoey's van after she opened the doors? ", "qid": 1, "show_name": "How I Met You Mother", "ts": "45.05-61.29", "vid_name": "met_s06e05_seg02_clip_09"}


# # NextQA base CSV
# video_id	frame_count	width	height	question	answer	qid	type	a0	a1	a2	a3	a4
# 4010069381	369	640	480	'how do the two man play the instrument'	0	6	CH	'roll the handle'	'tap their feet'	'strum the string'	'hit with sticks'	'pat with hand'

# # NextQA base JSON
# {"question": "how many children are in the video", "option_0": "one", "option_1": "three", "option_2": "seven", "option_3": "two", "option_4": "five", "answer": 3, "video": 3238737531}, {"question": "why is the blue sweater guy looking at the shirtless men", "option_0": "sharing with his friends", "option_1": "found the man funny", "option_2": "poor vision", "option_3": "keep hands warm", "option_4": "training", "answer": 4, "video": 8968804598}

# # NextQA CLIP/SIM caption
# video_id	frame_count	width	height	question	answer	qid	type	a0	a1	a2	a3	a4
# 3238737531	2303	640	480	'a little girl and a little boy standing in front of a house how many children are in the video'	3	2	DC	'one'	'three'	'seven'	'two'	'five'

