small_batch_size = 64
MAX_LENGTH = 50
MIN_COUNT = 3    # 阈值为3

HUANGJI_SOURCE = 'xiaohuangji50w_nofenci.conv'

# 配置模型
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 128

# 从哪个checkpoint恢复，如果是None，那么从头开始训练。
loadFilename = None
checkpoint_iter = 4000


# 配置训练的超参数和优化器 
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500
save_dir = 'check'
corpus_name = 'cornell_corpus'