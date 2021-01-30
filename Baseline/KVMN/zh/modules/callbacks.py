
from fastNLP import Callback, Tester, DataSet


class EvaluateCallback(Callback):
    """
    通过使用该Callback可以使得Trainer在evaluate dev之外还可以evaluate其它数据集，比如测试集。每一次验证dev之前都会先验证EvaluateCallback
    中的数据。
    """

    def __init__(self, data=None, tester=None,
                 use_knowledge=False,
                 knowledge_type=None,
                 pos_th=1,
                 dep_th=1,
                 chunk_th=1,
                 test_feature_data=None,
                 feature2count=None,
                 feature2id=None,
                 id2feature=None,
                 use_zen=False,
                 zen_model=None,
                 zen_dataset=None
                 ):
        """
        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用Trainer中的metric对数据进行验证。如果需要传入多个
            DataSet请通过dict的方式传入。
        :param ~fastNLP.Tester,Dict[~fastNLP.DataSet] tester: Tester对象, 通过使用Tester对象，可以使得验证的metric与Trainer中
            的metric不一样。
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        self.use_knowledge = use_knowledge
        self.knowledge_type = knowledge_type
        self.pos_th = pos_th
        self.dep_th = dep_th
        self.chunk_th = chunk_th
        self.test_feature_data = test_feature_data
        self.feature2count = feature2count
        self.feature2id = feature2id
        self.id2feature = id2feature
        self.use_zen = use_zen
        self.zen_model = zen_model
        self.zen_dataset = zen_dataset

        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

    def on_train_begin(self):
        if len(self.datasets) > 0 and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra DataSet to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics, verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm,
                                use_knowledge=self.use_knowledge,
                                knowledge_type=self.knowledge_type,
                                pos_th=self.pos_th,
                                dep_th=self.dep_th,
                                chunk_th=self.chunk_th,
                                test_feature_data=self.test_feature_data,
                                feature2count=self.feature2count,
                                feature2id=self.feature2id,
                                id2feature=self.id2feature,
                                use_zen=self.use_zen,
                                zen_model=self.zen_model,
                                zen_dataset=self.zen_dataset
                                )
                self.testers[key] = tester

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if len(self.testers) > 0:
            for idx, (key, tester) in enumerate(self.testers.items()):
                try:
                    eval_result = tester.test()
                    if idx == 0:
                        indicator, indicator_val = _check_eval_results(eval_result)
                        if indicator_val>self.best_test_metric_sofar:
                            self.best_test_metric_sofar = indicator_val
                            self.best_test_epoch = self.epoch
                            self.best_test_sofar = eval_result
                    if better_result:
                        self.best_dev_test = eval_result
                        self.best_dev_epoch = self.epoch
                    self.logger.info("EvaluateCallback evaluation on {}:".format(key))
                    self.logger.info(tester._format_eval_results(eval_result))
                except Exception as e:
                    self.logger.error("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        if self.best_test_sofar:
            self.logger.info("Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(self.best_test_sofar, self.best_test_epoch))
        if self.best_dev_test:
            self.logger.info("Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(self.best_dev_test, self.best_dev_epoch))


def _check_eval_results(metrics, metric_key=None):
    # metrics: tester返回的结果
    # metric_key: 一个用来做筛选的指标，来自Trainer的初始化
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]  # 取第一个metric

        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val
