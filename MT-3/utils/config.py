sample_rate = 16000
cycle_len = 240000 # 2.5 s * 4000 4 s * 4000
cycle_len_cough_heavy=13*16000# 5s
cycle_len_va=160000# 10s
win_length = 2048#256
hop_length = 512#128
mel_bins = 128

labels = [0,1]

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}
