import os, sys, time, math, re
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


root_dir = os.path.join(os.path.expanduser('~'), 'tfhub')
lang_models = { 'en': 'uncased_L-12_H-768_A-12',
				'cn': 'chinese_L-12_H-768_A-12' }
try: import bert
except: pass
class BERTLayer:
	def __init__(self, lang='en', adapter_size=None):
		model_name = lang_models[lang]
		model_dir = os.path.join(root_dir, model_name)
		self.bert_params = bert.params_from_pretrained_ckpt(model_dir)
		self.bert_params.adapter_size = adapter_size
		self.model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
		self.l_bert = bert.BertModelLayer.from_params(self.bert_params, name="bert")

		do_lower_case = not (model_name.find("cased") == 0 or model_name.find("multi_cased") == 0)
		bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, self.model_ckpt)
		vocab_file = os.path.join(model_dir, "vocab.txt")

		self.tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

	def get_bert_layer(self):
		return self.l_bert

	def post_build_model(self):
		bert.load_bert_weights(self.l_bert, self.model_ckpt) 

	def set_trainable(self, layer_num, obj=False):
		lst = [self.l_bert.embeddings_layer] + self.l_bert.encoders_layer.encoder_layers
		for x in lst[:layer_num]: x.trainable = obj

	def get_pooled_output(self, seq_output):
		return tf.keras.layers.Lambda(lambda x:x[:,0,:])(seq_output)

	def tokenize_sentence(self, sent):
		return self.tokenizer.tokenize(sent)

	def convert_single_sentence(self, seqA, max_seq_len, seqB=None, max_A_len=99999, return_seq_tokens=False):
		if type(seqA) is type(''): seqA = self.tokenize_sentence(seqA)
		tokens = ["[CLS]"]
		tokens += seqA
		if len(tokens)+1 >= max_seq_len: tokens = tokens[:max_seq_len-1]
		if len(tokens)+1 >= max_A_len: tokens = tokens[:max_A_len-1]
		tokens.append("[SEP]") 
		if seqB is not None:
			if type(seqB) is type(''): seqB = self.tokenize_sentence(seqB)
			tokens += seqB
			if len(tokens)+1 >= max_seq_len: tokens = tokens[:max_seq_len-1]
			tokens.append("[SEP]") 
		if return_seq_tokens:
			return {'tokens': tokens, 'seqA':seqA, 'seqB': seqB}
		return tokens

#try: import sentencepiece as spm
#except: print('ALBERT unavailable')
#class ALBERTLayer:
#	def __init__(self, lang='cn'):
#		model_name = "albert_large_zh"
#		self.albert_dir = bert.fetch_google_albert_model(model_name, root_dir)
#		model_params = bert.albert_params('albert_large')
#		model_params.shared_layer = False
#		self.l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
		
#		vocab_file = os.path.join(self.albert_dir, "vocab_chinese.txt")
#		self.tokenizer = bert.albert_tokenization.FullTokenizer(vocab_file=vocab_file)

#	def get_bert_layer(self):
#		return self.l_bert

#	def post_build_model(self):
#		model_ckpt = os.path.join(self.albert_dir, "model.ckpt-best")
#		bert.load_albert_weights(self.l_bert, model_ckpt)

#	def set_trainable(self, layer_num, obj=False):
#		lst = [self.l_bert.embeddings_layer] + self.l_bert.encoders_layer.encoder_layers
#		for x in lst[:layer_num]: x.trainable = obj

#	def get_pooled_output(self, seq_output):
#		return tf.keras.layers.Lambda(lambda x:x[:,0,:])(seq_output)

#	def tokenize_sentence(self, sent):
#		return self.tokenizer.tokenize(sent)

#	def convert_single_sentence(self, seqA, max_seq_len, seqB=None, max_A_len=99999, return_seq_tokens=False):
#		if type(seqA) is type(''): seqA = self.tokenize_sentence(seqA)
#		tokens = ["[CLS]"]
#		tokens += seqA
#		if len(tokens)+1 >= max_seq_len: tokens = tokens[:max_seq_len-1]
#		if len(tokens)+1 >= max_A_len: tokens = tokens[:max_A_len-1]
#		tokens.append("[SEP]") 
#		if seqB is not None:
#			if type(seqB) is type(''): seqB = self.tokenize_sentence(seqB)
#			tokens += seqB
#			if len(tokens)+1 >= max_seq_len: tokens = tokens[:max_seq_len-1]
#			tokens.append("[SEP]") 
#		if return_seq_tokens:
#			return {'tokens': tokens, 'seqA':seqA, 'seqB': seqB}
#		return tokens

		
def get_suggested_scheduler(init_lr=5e-5, total_steps=10000, warmup_ratio=0.1):
	opt_lr = K.variable(init_lr)
	warmup_steps = int(warmup_ratio * total_steps)
	warmup = WarmupCallback(opt_lr, init_lr, total_steps, warmup_steps)
	return warmup, opt_lr

class WarmupCallback(Callback):
	def __init__(self, lr_var, init_lr, total_steps, warmup_steps=0):
		self.step = 0
		self.lr_var = lr_var
		self.init_lr = init_lr
		self.warmup = warmup_steps
		self.total_steps = total_steps
	def on_batch_begin(self, batch, logs):
		self.step += 1
		if self.step <= self.warmup: 
			new_lr = self.init_lr * (self.step / self.warmup)
		else: 
			new_lr = self.init_lr * max(0, 1 - self.step / self.total_steps)
		K.set_value(self.lr_var, new_lr)
		K.set_value(self.model.optimizer.lr, new_lr)

def convert_single_setences(sens, maxlen, tokenizer, details=False):
	X = np.zeros((len(sens), maxlen), dtype='int32')
	datas = []
	for i, s in enumerate(sens):
		tokens = tokenizer.tokenize(s)[:maxlen-2]
		if details:
			otokens = restore_token_list(s, tokens)
			datas.append({'id':i, 's':s, 'otokens':otokens})
		tt = ['[CLS]'] + tokens + ['[SEP]']
		tids = tokenizer.convert_tokens_to_ids(tt)
		X[i,:len(tids)] = tids
	if details: return datas, X
	return X

def add_special_marks(tokens, maxlen):
	tt = ['[CLS]'] + tokens
	if len(tt)+1 > maxlen: tt = tt[:maxlen-1] 
	tt.append('[SEP]')
	return tt

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""
  def __init__(self,
               initial_learning_rate,
               decay_schedule_fn,
               warmup_steps,
               power=1.0,
               name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: self.decay_schedule_fn(step),
          name=name)

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }


def get_suggested_optimizer(init_lr=5e-5, total_steps=10000, warmup_ratio=0.1):
	lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
				initial_learning_rate=init_lr,
				decay_steps=total_steps,
				end_learning_rate=0)
	lr_schedule = WarmUp(initial_learning_rate=init_lr,
				decay_schedule_fn=lr_schedule,
				warmup_steps=int(total_steps*warmup_ratio))
	return AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01, exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])


class AdamWeightDecay(tf.keras.optimizers.Adam):
  """Adam enables L2 weight decay and clip_by_global_norm on gradients.
  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.
  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               weight_decay_rate=0.0,
               include_in_weight_decay=None,
               exclude_from_weight_decay=None,
               name='AdamWeightDecay',
               **kwargs):
    super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
    self.weight_decay_rate = weight_decay_rate
    self._include_in_weight_decay = include_in_weight_decay
    self._exclude_from_weight_decay = exclude_from_weight_decay

  @classmethod
  def from_config(cls, config):
    return super(AdamWeightDecay, cls).from_config(config)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
        self.weight_decay_rate, name='adam_weight_decay_rate')

  def _decay_weights_op(self, var, learning_rate, apply_state):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
      return var.assign_sub(
          learning_rate * var *
          apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
          use_locking=self._use_locking)
    return tf.no_op()

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    grads, tvars = list(zip(*grads_and_vars))
    if experimental_aggregate_gradients:
      # when experimental_aggregate_gradients = False, apply_gradients() no
      # longer implicitly allreduce gradients, users manually allreduce gradient
      # and passed the allreduced grads_and_vars. For now, the
      # clip_by_global_norm will be moved to before the explicit allreduce to
      # keep the math the same as TF 1 and pre TF 2.2 implementation.
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name, experimental_aggregate_gradients=experimental_aggregate_gradients)

  def _get_lr(self, var_device, var_dtype, apply_state):
    """Retrieves the learning rate with the given state."""
    if apply_state is None:
      return self._decayed_lr_t[var_dtype], {}

    apply_state = apply_state or {}
    coefficients = apply_state.get((var_device, var_dtype))
    if coefficients is None:
      coefficients = self._fallback_apply_state(var_device, var_dtype)
      apply_state[(var_device, var_dtype)] = coefficients

    return coefficients['lr_t'], dict(apply_state=apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

  def get_config(self):
    config = super(AdamWeightDecay, self).get_config()
    config.update({
        'weight_decay_rate': self.weight_decay_rate,
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.weight_decay_rate == 0:
      return False
    if self._include_in_weight_decay:
      for r in self._include_in_weight_decay:
        if re.search(r, param_name) is not None:
          return True
    if self._exclude_from_weight_decay:
      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True


## THESE FUNCTIONS ARE TESTED FOR CHS LANGUAGE ONLY
def gen_token_list_inv_pointer(sent, token_list):
	sent = sent.lower()
	otiis = []; iis = 0 
	for it, token in enumerate(token_list):
		otoken = token.lstrip('#')
		if token[0] == '[' and token[-1] == ']': otoken = ''
		niis = iis
		while niis < len(sent):
			if sent[niis:].startswith(otoken): break
			if otoken in '-"' and sent[niis][0] in '—“”': break
			niis += 1
		if niis >= len(sent): niis = iis
		otiis.append(niis)
		iis = niis + max(1, len(otoken))
	#for tt, ii in zip(token_list, otiis): print(tt, sent[ii:ii+len(tt)])
	#for i, iis in enumerate(otiis): 
	#	assert iis < len(sent)
	#	otoken = token_list[i].strip('#')
	#	assert otoken == '[UNK]' or sent[iis:iis+len(otoken)] == otoken
	return otiis

# restore [UNK] tokens to the original tokens
def restore_token_list(sent, token_list):
	invp = gen_token_list_inv_pointer(sent, token_list)
	invp.append(len(sent))
	otokens = [sent[u:v] for u,v in zip(invp, invp[1:])]
	processed = -1
	for ii, tk in enumerate(token_list):
		if tk != '[UNK]': continue
		if ii < processed: continue
		for jj in range(ii+1, len(token_list)):
			if token_list[jj] != '[UNK]': break
		else: jj = len(token_list)
		allseg = sent[invp[ii]:invp[jj]]

		if ii + 1 == jj: continue
		seppts = [0] + [i for i, x in enumerate(allseg) if i > 0 and i+1 < len(allseg) and x == ' ' and allseg[i-1] != ' ']
		if allseg[seppts[-1]:].replace(' ', '') == '': seppts = seppts[:-1]
		seppts.append(len(allseg))
		if len(seppts) == jj - ii + 1:
			for k, (u,v) in enumerate(zip(seppts, seppts[1:])): 
				otokens[ii+k] = allseg[u:v]
		processed = jj + 1
	if invp[0] > 0: otokens[0] = sent[:invp[0]] + otokens[0]
	if ''.join(otokens) != sent:
		raise Exception('restore tokens failed, text and restored:\n%s\n%s' % (sent, ''.join(otokens)))
	return otokens

def gen_word_level_labels(sent, token_list, word_list, pos_list=None):
	otiis = gen_token_list_inv_pointer(sent, token_list)
	wdiis = [];	iis = 0
	for ip, pword in enumerate(word_list):
		niis = iis
		while niis < len(sent):
			if pword == '' or sent[niis:].startswith(pword[0]): break
			niis += 1
		wdiis.append(niis)
		iis = niis + len(pword)
	#for tt, ii in zip(word_list, wdiis): print(tt, sent[ii:ii+len(tt)])

	rlist = [];	ip = 0
	for it, iis in enumerate(otiis):
		while ip + 1 < len(wdiis) and wdiis[ip+1] <= iis: ip += 1
		if iis == wdiis[ip]: rr = 'B'
		elif iis > wdiis[ip]: rr = 'I'
		rr += '-' + pos_list[ip]
		rlist.append(rr)
	#for rr, tt in zip(rlist, token_list): print(rr, tt)
	return rlist

def normalize_sentence(text):
	text = re.sub('[“”]', '"', text)
	text = re.sub('[—]', '-', text)
	text = re.sub('[^\u0000-\u007f\u4e00-\u9fa5\u3001-\u303f\uff00-\uffef·—]', ' \u2800 ', text)
	return text

if __name__ == '__main__':
	from transformers import BertTokenizer
	tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
	sent = '6月13日起，法国ARTE频道将推出一部12集的新迷你剧《奥德修斯》(Odysseus)，是编剧Frédéric Azémar用更自由的视角对荷马史诗的一次改编和延续'
	tokens = tokenizer.tokenize(sent)
	otokens = restore_token_list(sent, tokens)
	print(tokens)
	print(otokens)
	print('done')