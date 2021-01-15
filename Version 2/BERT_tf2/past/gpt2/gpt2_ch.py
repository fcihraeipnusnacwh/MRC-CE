import os, sys, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras.models import Model
from slayers import *

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))
keras.utils.get_custom_objects()['gelu'] = gelu

class Transformer(object):
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
		self.layers = {} if layers is None else layers
		self.name = name
		self.built = False

	def build(self, layer_norm_cond=None, layer_norm_cond_hidden_size=None, layer_norm_cond_hidden_act=None, additional_input_layers=None, **kwargs):
		"""模型构建函数
		layer_norm_*系列参数为实现Conditional Layer Normalization时使用，
		用来实现以“固定长度向量”为条件的条件Bert。
		"""
		if self.built:
			return None
		# Input
		inputs = self.get_inputs()
		self.set_inputs(inputs, additional_input_layers)
		# Other
		self.layer_norm_conds = [
			layer_norm_cond,
			layer_norm_cond_hidden_size,
			layer_norm_cond_hidden_act or 'linear',
		]
		# Call
		outputs = self.call(inputs)
		self.set_outputs(outputs)
		# Model
		self.model = Model(self.inputs, self.outputs, name=self.name)
		self.built = True

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
		if name not in self.layers:
			layer = layer(**kwargs)
			name = layer.name
			self.layers[name] = layer

		return self.layers[name](inputs, **arguments)

	def get_inputs(self):
		raise NotImplementedError

	def apply_embeddings(self, inputs):
		raise NotImplementedError

	def apply_main_layers(self, inputs, index):
		raise NotImplementedError

	def apply_final_layers(self, inputs):
		raise NotImplementedError

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
		mapping = {k: v for k, v in mapping.items() if k in self.layers}

		weight_value_pairs = []
		for layer, variables in mapping.items():
			layer = self.layers[layer]
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
				layer = self.layers[layer]
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

	def get_inputs(self):
		"""GPT2_ML的输入是token_ids和segment_ids
		"""
		x_in = Input(shape=(self.sequence_length,), name='Input-Token')
		return x_in

	def apply_embeddings(self, inputs):
		"""GPT2_ML的embedding是token、position两者embedding之和
		"""
		x = inputs
		z = self.layer_norm_conds[0]

		x = self.apply(
			inputs=x,
			layer=Embedding,
			input_dim=self.vocab_size,
			output_dim=self.embedding_size,
			embeddings_initializer=self.initializer,
			mask_zero=True,
			name='Embedding-Token'
		)
		x = self.apply(
			inputs=x,
			layer=PositionEmbedding,
			input_dim=self.max_position,
			output_dim=self.embedding_size,
			merge_mode='add',
			embeddings_initializer=self.initializer,
			name='Embedding-Position'
		)
		x = self.apply(
			inputs=self.simplify([x, z]),
			layer=LayerNormalization,
			epsilon=1e-5,
			conditional=(z is not None),
			hidden_units=self.layer_norm_conds[1],
			hidden_activation=self.layer_norm_conds[2],
			hidden_initializer=self.initializer,
			name='Embedding-Norm'
		)
		if self.embedding_size != self.hidden_size:
			x = self.apply(
				inputs=x,
				layer=Dense,
				units=self.hidden_size,
				kernel_initializer=self.initializer,
				name='Embedding-Mapping'
			)

		return x

	def apply_main_layers(self, inputs, index, lasts=None):
		"""GPT2_ML的主体是基于Self-Attention的模块
		顺序：Att  --> LN --> FFN --> Add --> LN
		"""
		x = inputs
		z = self.layer_norm_conds[0]

		attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
		feed_forward_name = 'Transformer-%d-FeedForward' % index
		attention_mask = self.compute_attention_mask(index)

		# Self Attention
		if lasts is None:
			xi, x, arguments = x, [x, x, x, attention_mask], {'a_mask': True}
		else:
			ll = K.concatenate([lasts, x], axis=1)
			xi, x, arguments = x, [x, ll, ll, attention_mask], {'a_mask': True}
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
			inputs=self.simplify([x, z]),
			layer=LayerNormalization,
			epsilon=1e-5,
			conditional=(z is not None),
			hidden_units=self.layer_norm_conds[1],
			hidden_activation=self.layer_norm_conds[2],
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
			inputs=self.simplify([x, z]),
			layer=LayerNormalization,
			epsilon=1e-5,
			conditional=(z is not None),
			hidden_units=self.layer_norm_conds[1],
			hidden_activation=self.layer_norm_conds[2],
			hidden_initializer=self.initializer,
			name='%s-Norm-1' % feed_forward_name
		)

		return x

	def apply_final_layers(self, inputs):
		"""剩余部分
		"""
		x = inputs
		z = self.layer_norm_conds[0]

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
	def build(self, layer_norm_cond=None, layer_norm_cond_hidden_size=None, layer_norm_cond_hidden_act=None, additional_input_layers=None, **kwargs):
		if self.built: return None
		# Input
		inputs = self.get_inputs()
		self.set_inputs(inputs, additional_input_layers)
		# Other
		self.layer_norm_conds = [
			layer_norm_cond,
			layer_norm_cond_hidden_size,
			layer_norm_cond_hidden_act or 'linear',
		]
		# Call
		lasts = [Input(shape=(None, self.hidden_size)) for _ in range(self.num_hidden_layers)]

		outputs = self.call(inputs)
		outputs_lm = self.call(inputs, lasts)
		self.set_outputs(outputs)
		# Model
		self.model = Model(self.inputs, self.outputs, name=self.name)
		self.model_lm = Model([self.inputs]+lasts, outputs_lm)
		self.built = True

	def apply_embeddings(self, inputs, lasts=None):
		"""GPT2_ML的embedding是token、position两者embedding之和
		"""
		x = inputs
		z = self.layer_norm_conds[0]
		xi = inputs

		x = self.apply(
			inputs=x,
			layer=Embedding,
			input_dim=self.vocab_size,
			output_dim=self.embedding_size,
			embeddings_initializer=self.initializer,
			mask_zero=True,
			name='Embedding-Token'
		)
		if lasts is None:
			x_pos = K.cumsum(K.ones_like(xi), 1) - 1
		else:
			x_pos = K.sum(K.ones_like(K.sum(lasts[0], -1), dtype='int32'), -1)
		x = self.apply(
			inputs=[x, x_pos],
			layer=PositionEmbedding,
			custom_position_ids=True,
			input_dim=self.max_position,
			output_dim=self.embedding_size,
			merge_mode='add',
			embeddings_initializer=self.initializer,
			name='Embedding-Position'
		)
		x = self.apply(
			inputs=self.simplify([x, z]),
			layer=LayerNormalization,
			epsilon=1e-5,
			conditional=(z is not None),
			hidden_units=self.layer_norm_conds[1],
			hidden_activation=self.layer_norm_conds[2],
			hidden_initializer=self.initializer,
			name='Embedding-Norm'
		)
		if self.embedding_size != self.hidden_size:
			x = self.apply(
				inputs=x,
				layer=Dense,
				units=self.hidden_size,
				kernel_initializer=self.initializer,
				name='Embedding-Mapping'
			)

		return x

	def call(self, inputs, lasts=None):
		allouts = []
		outputs = self.apply_embeddings(inputs, lasts)
		allouts.append(outputs)
		for i in range(self.num_hidden_layers):
			if lasts is not None:
				outputs = self.apply_main_layers(outputs, i, lasts[i])
			else: 
				outputs = self.apply_main_layers(outputs, i)
			allouts.append(outputs)
		outputs = self.apply_final_layers(outputs)
		return [outputs] + allouts


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

#transformer = GPT2_ML_LMHead(**configs)
transformer = GPT2_ML(**configs)
transformer.build(**configs)
transformer.load_weights_from_checkpoint(checkpoint_path)
print('load ok')

model = transformer.model

class AutoRegressiveDecoder(object):
	"""通用自回归生成模型解码基类
	包含beam search和random sample两种策略
	"""
	def __init__(self, start_id, end_id, maxlen, minlen=None):
		self.start_id = start_id
		self.end_id = end_id
		self.maxlen = maxlen
		self.minlen = minlen or 1
		if start_id is None:
			self.first_output_ids = np.empty((1, 0), dtype=int)
		else:
			self.first_output_ids = np.array([[self.start_id]])

	@staticmethod
	def set_rtype(default='probas'):
		"""用来给predict方法加上rtype参数，并作相应的处理
		"""
		def actual_decorator(predict):
			def new_predict(self, inputs, output_ids, step, rtype=default):
				assert rtype in ['probas', 'logits']
				result = predict(self, inputs, output_ids, step)
				if default == 'probas':
					if rtype == 'probas':
						return result
					else:
						return np.log(result + 1e-12)
				else:
					if rtype == 'probas':
						return softmax(result, -1)
					else:
						return result

			return new_predict

		return actual_decorator

	def predict(self, inputs, output_ids, step, rtype='logits'):
		"""用户需自定义递归预测函数
		rtype为字符串logits或probas，用户定义的时候，应当根据rtype来
		返回不同的结果，rtype=probas时返回归一化的概率，rtype=logits时
		则返回softmax前的结果或者概率对数。
		"""
		raise NotImplementedError

	def beam_search(self, inputs, topk, min_ends=1):
		"""beam search解码
		说明：这里的topk即beam size；
		返回：最优解码序列。
		"""
		inputs = [np.array([i]) for i in inputs]
		output_ids, output_scores = self.first_output_ids, np.zeros(1)
		for step in range(self.maxlen):
			scores = self.predict(inputs, output_ids, step, 'logits')  # 计算当前得分
			if step == 0:  # 第1步预测后将输入重复topk次
				inputs = [np.repeat(i, topk, axis=0) for i in inputs]
			scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
			indices = scores.argpartition(-topk, axis=None)[-topk:]  # 仅保留topk
			indices_1 = indices // scores.shape[1]  # 行索引
			indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
			output_ids = np.concatenate([output_ids[indices_1], indices_2], 1)  # 更新输出
			output_scores = np.take_along_axis(scores, indices, axis=None)  # 更新得分
			end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
			if output_ids.shape[1] >= self.minlen:  # 最短长度判断
				best_one = output_scores.argmax()  # 得分最大的那个
				if end_counts[best_one] == min_ends:  # 如果已经终止
					return output_ids[best_one]  # 直接输出
				else:  # 否则，只保留未完成部分
					flag = (end_counts < min_ends)  # 标记未完成序列
					if not flag.all():  # 如果有已完成的
						inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
						output_ids = output_ids[flag]  # 扔掉已完成序列
						output_scores = output_scores[flag]  # 扔掉已完成序列
						end_counts = end_counts[flag]  # 扔掉已完成end计数
						topk = flag.sum()  # topk相应变化
		# 达到长度直接输出
		return output_ids[output_scores.argmax()]

	def random_sample(self, inputs, n, topk=None, topp=None, min_ends=1):
		"""随机采样n个结果
		说明：非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp
			 表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
		返回：n个解码序列组成的list。
		"""
		inputs = [np.array([i]) for i in inputs]
		output_ids = self.first_output_ids
		results = []
		for step in range(self.maxlen):
			probas = self.predict(inputs, output_ids, step, 'probas')  # 计算当前概率
			probas /= probas.sum(axis=1, keepdims=True)  # 确保归一化
			if step == 0:  # 第1步预测后将结果重复n次
				probas = np.repeat(probas, n, axis=0)
				inputs = [np.repeat(i, n, axis=0) for i in inputs]
				output_ids = np.repeat(output_ids, n, axis=0)
			if topk is not None:
				k_indices = probas.argpartition(-topk, axis=1)[:, -topk:]  # 仅保留topk
				probas = np.take_along_axis(probas, k_indices, axis=1)  # topk概率
				probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
			if topp is not None:
				p_indices = probas.argsort(axis=1)[:, ::-1]  # 从高到低排序
				probas = np.take_along_axis(probas, p_indices, axis=1)  # 排序概率
				cumsum_probas = np.cumsum(probas, axis=1)  # 累积概率
				flag = np.roll(cumsum_probas >= topp, 1, axis=1)  # 标记超过topp的部分
				flag[:, 0] = False  # 结合上面的np.roll，实现平移一位的效果
				probas[flag] = 0  # 后面的全部置零
				probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
			sample_func = lambda p: np.random.choice(len(p), p=p)  # 按概率采样函数
			sample_ids = np.apply_along_axis(sample_func, 1, probas)  # 执行采样
			sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
			if topp is not None:
				sample_ids = np.take_along_axis(p_indices, sample_ids, axis=1)  # 对齐原id
			if topk is not None:
				sample_ids = np.take_along_axis(k_indices, sample_ids, axis=1)  # 对齐原id
			output_ids = np.concatenate([output_ids, sample_ids], 1)  # 更新输出
			end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
			if output_ids.shape[1] >= self.minlen:  # 最短长度判断
				flag = (end_counts == min_ends)  # 标记已完成序列
				if flag.any():  # 如果有已完成的
					for ids in output_ids[flag]:  # 存好已完成序列
						results.append(ids)
					flag = (flag == False)  # 标记未完成序列
					inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
					output_ids = output_ids[flag]  # 只保留未完成部分候选集
					end_counts = end_counts[flag]  # 只保留未完成部分end计数
					if len(output_ids) == 0:
						break
		# 如果还有未完成序列，直接放入结果
		for ids in output_ids:
			results.append(ids)
		# 返回结果
		return results

class ArticleCompletion(AutoRegressiveDecoder):
	"""基于随机采样的文章续写
	"""
	@AutoRegressiveDecoder.set_rtype('probas')
	def predict(self, inputs, output_ids, step):
		token_ids = np.concatenate([inputs[0], output_ids], 1)
		return model.predict_on_batch(token_ids)[:, -1]

	def generate(self, text, n=1, topk=5):
		token_ids, _ = tokenizer.encode(text)
		results = self.random_sample([token_ids], n, topk)  # 基于随机采样
		return [text + tokenizer.decode(ids) for ids in results]

writer = ArticleCompletion(start_id=None, end_id=511, maxlen=75, minlen=50)
tic = time.process_time()
print(writer.generate('太阳从哪边升起呢？事实上，太阳从', topk=2))
print(writer.generate('太阳从东边还是西边升起？事实上，太阳从', topk=2))
print(writer.generate('猫有多少条腿？事实上，猫有', topk=2))
print(writer.generate('猫有多少条腿？事实上，猫有', topk=2))
print(writer.generate('狗有多少条腿？事实上，狗有', topk=2))
print('%.3fs' % (time.process_time() - tic))

sys.exit()
token_ids, _ = tokenizer.encode('评估准确率的话') 
token_ids = np.array([token_ids])

ret = model.predict_on_batch(token_ids)
outs, lasts = ret[0], ret[1:]
nextid = np.argmax(outs[0,-1,:])
print(nextid, tokenizer.decode([nextid]))

zz = transformer.model_lm.predict_on_batch([np.array([[8024]])] + lasts)
for i, x in enumerate(zz[1:]): 
	lasts[i] = np.concatenate([lasts[i], x], axis=1)
nextid = np.argmax(zz[0][0,-1,:])
print(nextid, tokenizer.decode([nextid]))

token_ids, _ = tokenizer.encode('评估准确率的话，') 
token_ids = np.array([token_ids])
ret2 = model(token_ids)

print(zz[2][:,-1])
print(ret2[2][:,-1])


def sample_decode(sr, topk=5, end_id=511, maxlen=50, minlen=50):
	token_ids, _ = tokenizer.encode(sr) 
	token_ids = np.array([token_ids])
	outs = transformer.model.predict_on_batch(token_ids)
	outs, lasts = outs[0], outs[1:]
	rets = []
	for ii in range(maxlen):
		probas = outs[:,-1,:]
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
		#r = np.argmax(outs[0,-1,:])
		rets.append(r)
		if r == end_id and len(rets) > minlen: break
		#tic = time.process_time()
		outs = transformer.model_lm.predict_on_batch([np.array([[r]])] + lasts)	
		#print('%.3fs' % (time.process_time() - tic))
		for i, x in enumerate(outs[1:]):
			lasts[i] = K.concatenate([lasts[i], x], axis=1)
		outs = outs[0]
	return rets

tic = time.process_time()
rr = sample_decode('夏天的风', topk=2, maxlen=50, minlen=50)
rr = '夏天的风' + tokenizer.decode(rr)
print(rr)
print('%.3fs' % (time.process_time() - tic))
