import os, sys, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from slayers import *

def gelu(x):
	return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))
keras.utils.get_custom_objects()['gelu'] = gelu

class Transformer(Model):
	def __init__(
		self,
		vocab_size,  # 词表大小
		hidden_size,  # 编码维度
		num_hidden_layers,  # Transformer总层数
		num_attention_heads,  # Attention的头数
		intermediate_size,  # FeedForward的隐层维度
		hidden_act,  # FeedForward隐层的激活函数
		dropout_rate=None,  # Dropout比例
		embedding_size=None,  # 是否指定embedding_size
		attention_key_size=None,  # Attention中Q,K的head_size
		sequence_length=None,  # 是否固定序列长度
		keep_tokens=None,  # 要保留的词ID列表
		layers=None,  # 外部传入的Keras层
		name=None,  # 模型名称
		**kwargs
	):
		super(Transformer, self).__init__()
		if keep_tokens is None:
			self.vocab_size = vocab_size
		else:
			self.vocab_size = len(keep_tokens)
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.attention_head_size = hidden_size // num_attention_heads
		self.attention_key_size = attention_key_size or self.attention_head_size
		self.intermediate_size = intermediate_size
		self.dropout_rate = dropout_rate or 0
		self.hidden_act = hidden_act
		self.embedding_size = embedding_size or hidden_size
		self.sequence_length = sequence_length
		self.keep_tokens = keep_tokens
		self.attention_mask = None
		self.position_bias = None
		self.llayers = {} if layers is None else layers
		self.built = False

	def call(self, inputs):
		"""定义模型的执行流程
		"""
		# Embedding
		outputs = self.apply_embeddings(inputs)
		# Main
		for i in range(self.num_hidden_layers):
			outputs = self.apply_main_layers(outputs, i)
		# Final
		outputs = self.apply_final_layers(outputs)
		return outputs

	def apply(self, inputs, layer=None, arguments=None, **kwargs):
		"""通过apply调用层会自动重用同名层
		inputs: 上一层的输出；
		layer: 要调用的层类名；
		arguments: 传递给layer.call的参数；
		kwargs: 传递给层初始化的参数。
		"""
		if layer is Dropout and self.dropout_rate == 0:
			return inputs

		arguments = arguments or {}
		name = kwargs.get('name')
		if name not in self.llayers:
			layer = layer(**kwargs)
			name = layer.name
			self.llayers[name] = layer

		return self.llayers[name](inputs, **arguments)

	def compute_attention_mask(self, inputs=None):
		"""定义每一层的Attention Mask
		"""
		return self.attention_mask

	def compute_position_bias(self, inputs=None):
		"""定义每一层的Position Bias（一般相对位置编码用）
		"""
		return self.position_bias

	def set_inputs(self, inputs, additional_input_layers=None):
		"""设置input和inputs属性
		"""
		if inputs is None:
			inputs = []
		elif not isinstance(inputs, list):
			inputs = [inputs]

		inputs = inputs[:]
		if additional_input_layers is not None:
			if not isinstance(additional_input_layers, list):
				additional_input_layers = [additional_input_layers]
			inputs.extend(additional_input_layers)

		self.inputs = inputs
		if len(inputs) > 1:
			self.input = inputs
		else:
			self.input = inputs[0]

	def set_outputs(self, outputs):
		"""设置output和oututs属性
		"""
		if not isinstance(outputs, list):
			outputs = [outputs]

		outputs = outputs[:]
		self.outputs = outputs
		if len(outputs) > 1:
			self.output = outputs
		else:
			self.output = outputs[0]

	@property
	def initializer(self):
		"""默认使用截断正态分布初始化
		"""
		return keras.initializers.TruncatedNormal(stddev=0.02)

	def simplify(self, inputs):
		"""将list中的None过滤掉
		"""
		inputs = [i for i in inputs if i is not None]
		if len(inputs) == 1:
			inputs = inputs[0]

		return inputs

	def load_variable(self, checkpoint, name):
		"""加载单个变量的函数
		"""
		return tf.train.load_variable(checkpoint, name)

	def create_variable(self, name, value):
		"""在tensorflow中创建一个变量
		"""
		return tf.Variable(value, name=name)

	def variable_mapping(self):
		"""构建keras层与checkpoint的变量名之间的映射表
		"""
		return {}

	def load_weights_from_checkpoint(self, checkpoint, mapping=None):
		"""根据mapping从checkpoint加载权重
		"""
		mapping = mapping or self.variable_mapping()
		mapping = {k: v for k, v in mapping.items() if k in self.llayers}

		weight_value_pairs = []
		for layer, variables in mapping.items():
			layer = self.llayers[layer]
			weights = layer.trainable_weights
			values = [self.load_variable(checkpoint, v) for v in variables]
			if isinstance(layer, MultiHeadAttention):
				"""如果key_size不等于head_size，则可以通过
				正交矩阵将相应的权重投影到合适的shape。
				"""
				count = 2
				if layer.use_bias:
					count += 2
				heads = self.num_attention_heads
				head_size = self.attention_head_size
				key_size = self.attention_key_size
				W = np.linalg.qr(np.random.randn(key_size, head_size))[0].T
				if layer.attention_scale:
					W = W * key_size**0.25 / head_size**0.25
				for i in range(count):
					w, v = weights[i], values[i]
					w_shape, v_shape = K.int_shape(w), v.shape
					if w_shape[-1] != v_shape[-1]:
						pre_shape = w_shape[:-1]
						v = v.reshape(pre_shape + (heads, head_size))
						v = np.dot(v, W)
						v = v.reshape(pre_shape + (heads * key_size,))
						values[i] = v

			weight_value_pairs.extend(zip(weights, values))
		K.batch_set_value(weight_value_pairs)

	def save_weights_as_checkpoint(self, filename, mapping=None):
		"""根据mapping将权重保存为checkpoint格式
		"""
		mapping = mapping or self.variable_mapping()
		mapping = {k: v for k, v in mapping.items() if k in self.layers}
		with tf.Graph().as_default():
			for layer, variables in mapping.items():
				layer = self.llayers[layer]
				values = K.batch_get_value(layer.trainable_weights)
				for name, value in zip(variables, values):
					self.create_variable(name, value)
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				saver = tf.train.Saver()
				saver.save(sess, filename, write_meta_graph=False)

class GPT2_ML(Transformer):
	"""构建GPT2_ML模型
	链接: https://github.com/imcaspar/gpt2-ml
	"""
	def __init__(
		self,
		max_position,  # 序列最大长度
		final_activation='softmax',  # 预测分布的激活函数
		**kwargs  # 其余参数
	):
		super(GPT2_ML, self).__init__(**kwargs)
		self.max_position = max_position
		self.final_activation = final_activation

	def apply_final_layers(self, inputs):
		"""剩余部分
		"""
		x = inputs

		# Language Model部分
		x = self.apply(
			inputs=x,
			layer=Embedding,
			arguments={'mode': 'dense'},
			name='Embedding-Token'
		)
		x = self.apply(
			inputs=x,
			layer=Activation,
			activation=self.final_activation,
			name='LM-Activation'
		)

		return x

	def load_variable(self, checkpoint, name):
		"""加载单个变量的函数
		"""
		variable = super(GPT2_ML, self).load_variable(checkpoint, name)
		if name == 'newslm/embeddings/word_embed':
			if self.keep_tokens is None:
				return variable
			else:
				return variable[self.keep_tokens]
		else:
			return variable

	def compute_attention_mask(self, inputs=None):
		"""添加下三角形式的attention mask
		"""
		if self.attention_mask is None:

			def lm_mask(s):
				seq_len = K.shape(s)[1]
				idxs = K.arange(0, seq_len)
				mask = idxs[None, :] <= idxs[:, None]
				mask = K.cast(mask, K.floatx())
				return mask[None, None]

			self.attention_mask = self.apply(
				inputs=self.inputs[0],
				layer=Lambda,
				function=lm_mask,
				name='Attention-LM-Mask'
			)

		return self.attention_mask

	def variable_mapping(self):
		"""映射到官方GPT2_ML权重格式
		"""
		mapping = {
			'Embedding-Token': ['newslm/embeddings/word_embed'],
			'Embedding-Position': ['newslm/embeddings/pos_embed'],
			'Embedding-Norm': [
				'newslm/embeddings/LayerNorm_embed_norm/beta',
				'newslm/embeddings/LayerNorm_embed_norm/gamma',
			],
		}

		for i in range(self.num_hidden_layers):
			prefix = 'newslm/layer%02d/' % i
			mapping.update({
				'Transformer-%d-MultiHeadSelfAttention' % i: [
					prefix + 'query_layer/kernel',
					prefix + 'query_layer/bias',
					prefix + 'key_layer/kernel',
					prefix + 'key_layer/bias',
					prefix + 'value_layer/kernel',
					prefix + 'value_layer/bias',
					prefix + 'context_projection_layer/kernel',
					prefix + 'context_projection_layer/bias',
				],
				'Transformer-%d-FeedForward-Norm-0' % i: [
					prefix + 'LayerNorm_mlp_ln0/beta',
					prefix + 'LayerNorm_mlp_ln0/gamma',
				],
				'Transformer-%d-FeedForward' % i: [
					prefix + 'intermediate/kernel',
					prefix + 'intermediate/bias',
					prefix + 'output/kernel',
					prefix + 'output/bias',
				],
				'Transformer-%d-FeedForward-Norm-1' % i: [
					prefix + 'LayerNorm_mlp_ln1/beta',
					prefix + 'LayerNorm_mlp_ln1/gamma',
				],
			})

		return mapping


class GPT2_ML_LMHead(GPT2_ML):
	def apply_main_layers(self, inputs, index, past=None):
		"""GPT2_ML的主体是基于Self-Attention的模块
		顺序：Att  --> LN --> FFN --> Add --> LN
		"""
		x = inputs

		attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
		feed_forward_name = 'Transformer-%d-FeedForward' % index

		# Self Attention
		if past is None: present = x
		else: present = tf.concat([past, x], axis=-2)

		def lm_mask(q, k):
			qlen, klen = K.shape(q)[1], K.shape(k)[1]
			idxq, idxk = K.arange(0, qlen), K.arange(0, klen)
			mask = (qlen-klen) + idxk[None, :] <= idxq[:, None]
			mask = K.cast(mask, K.floatx())
			return mask[None, None]

		attention_mask = lm_mask(x, present)
		xi, x, arguments = x, [x, present, present, attention_mask], {'a_mask': True}
		# the layer is reused, so must set attention_mask=1 to remove mask

		x = self.apply(
			inputs=x,
			layer=MultiHeadAttention,
			arguments=arguments,
			heads=self.num_attention_heads,
			head_size=self.attention_head_size,
			key_size=self.attention_key_size,
			kernel_initializer=self.initializer,
			name=attention_name
		)
		x = self.apply(
			inputs=x,
			layer=Dropout,
			rate=self.dropout_rate,
			name='%s-Dropout' % attention_name
		)
		x = self.apply(
			inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
		)

		# Feed Forward
		xi = x
		x = self.apply(
			inputs=x,
			layer=LayerNormalization,
			epsilon=1e-5,
			hidden_initializer=self.initializer,
			name='%s-Norm-0' % feed_forward_name
		)
		x = self.apply(
			inputs=x,
			layer=FeedForward,
			units=self.intermediate_size,
			activation=self.hidden_act,
			kernel_initializer=self.initializer,
			name=feed_forward_name
		)
		x = self.apply(
			inputs=x,
			layer=Dropout,
			rate=self.dropout_rate,
			name='%s-Dropout' % feed_forward_name
		)
		x = self.apply(
			inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
		)
		x = self.apply(
			inputs=x,
			layer=LayerNormalization,
			epsilon=1e-5,
			hidden_initializer=self.initializer,
			name='%s-Norm-1' % feed_forward_name
		)
		return x, present

	def apply_embeddings(self, inputs, position_ids=None):
		x = inputs;  xi = inputs

		x = self.apply(
			inputs=x,
			layer=Embedding,
			input_dim=self.vocab_size,
			output_dim=self.embedding_size,
			embeddings_initializer=self.initializer,
			mask_zero=True,
			name='Embedding-Token'
		)
		if position_ids is None:
			position_ids = K.cumsum(K.ones_like(xi), 1) - 1
		x = self.apply(
			inputs=[x, position_ids],
			layer=PositionEmbedding,
			custom_position_ids=True,
			input_dim=self.max_position,
			output_dim=self.embedding_size,
			merge_mode='add',
			embeddings_initializer=self.initializer,
			name='Embedding-Position'
		)
		x = self.apply(
			inputs=x,
			layer=LayerNormalization,
			epsilon=1e-5,
			hidden_initializer=self.initializer,
			name='Embedding-Norm'
		)
		return x

	def call(self, inputs, past=None, **kwargs):
		if isinstance(inputs, (tuple, list)):
			input_ids = inputs[0]
			past = inputs[1] if len(inputs) > 1 else past
		else:
			input_ids = inputs

		if past is None:
			past_length = 0
			past = [None] * self.num_hidden_layers
		else:
			past_length = tf.shape(past[0])[-2]

		position_ids = tf.range(past_length, tf.shape(input_ids)[-1]+past_length, dtype=tf.int32)[tf.newaxis, :]

		presents = ()
		output = self.apply_embeddings(inputs, position_ids)

		for i in range(self.num_hidden_layers):
			output, present = self.apply_main_layers(output, i, past[i])
			presents = presents + (present,)

		lm_logits = self.apply_final_layers(output)

		outputs = (lm_logits, presents)
		return outputs  # lm_logits, presents, (all hidden_states), (attentions)



dict_path = 'Chinese-GPT2_ML-1.5B-v1/vocab.txt'
config_path = 'Chinese-GPT2_ML-1.5B-v1/config.json'
checkpoint_path = 'Chinese-GPT2_ML-1.5B-v1/model.ckpt-100000'

from stokenizers import Tokenizer
tokenizer = Tokenizer(dict_path, token_start=None, token_end=None, do_lower_case=True)

configs = {}
if config_path is not None: configs.update(json.load(open(config_path)))
if 'max_position' not in configs:
	configs['max_position'] = configs.get('max_position_embeddings')
if 'dropout_rate' not in configs:
	configs['dropout_rate'] = configs.get('hidden_dropout_prob')

#model = TFGPT2LMHeadModel(configs)

transformer = GPT2_ML_LMHead(**configs)
zz = transformer(np.ones((1,1), dtype='int32')) # build the model
transformer.load_weights_from_checkpoint(checkpoint_path)
print('load ok')

token_ids, _ = tokenizer.encode('雨下整夜，我的爱溢出就像雨水') 
token_ids = np.array([token_ids])
ret = transformer(token_ids)

tic = time.process_time()
for ii in range(5):
	outs, pasts = ret[0], ret[1]
	nextid = np.argmax(outs[0,-1,:])
	print(nextid, tokenizer.decode([nextid]))
	ret = transformer(np.array([[nextid]]), pasts)
print('%.3fs' % (time.process_time() - tic))

def sample_decode(sr, topk=5, end_id=511, maxlen=50, minlen=50):
	token_ids, _ = tokenizer.encode(sr) 
	token_ids = np.array([token_ids])
	outs = transformer(token_ids)
	outs, pasts = outs[0], outs[1]
	rets = []
	for ii in range(maxlen):
		probas = outs[:,-1,:].numpy()
		probas /= probas.sum(axis=1, keepdims=True)  # 确保归一化
		if topk is not None:
			k_indices = probas.argpartition(-topk, axis=1)[:, -topk:]  # 仅保留topk
			probas = np.take_along_axis(probas, k_indices, axis=1)  # topk概率
			probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
		sample_func = lambda p: np.random.choice(len(p), p=p)  # 按概率采样函数
		sample_ids = np.apply_along_axis(sample_func, 1, probas)  # 执行采样
		sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
		if topk is not None:
			sample_ids = np.take_along_axis(k_indices, sample_ids, axis=1)  # 对齐原id
		r = sample_ids[0][0]
		rets.append(r)
		if r == end_id and len(rets) > minlen: break
		outs = transformer(np.array([[r]]), pasts)	
		outs, pasts = outs[0], outs[1]
	return rets

tic = time.process_time()
pre = '天堂有路你不走，'
rr = sample_decode(pre, topk=5, maxlen=50, minlen=50)
rr = pre + tokenizer.decode(rr)
print(rr)
print('%.3fs' % (time.process_time() - tic))
