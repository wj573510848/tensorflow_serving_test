# 简介

本项目的目的：

1. 探索使用tensorflow serving的方法
2. 测试tensorflow serving的各项性能

# 环境

ubuntu 16.04

使用conda安装

```shell
conda env create --file=environment.yml
```


# tensorflow serving 安装

详见官方教程：`https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md`

# 性能对比简介

* 使用text_cnn作为基准的神经网络模型。

* 将会搭建两个模型用于比较，一个使用tensorflow serving，另一个使用gunicorn+flask。

## 模型1-tensorflow serving

* 使用`release.py` 生成tensorflow-serving所需的模型文件

    ```shell
    python release.py
    ```

    将会在release目录下生成模型文件，目录结构为：`模型名 --> 版本 --> pb与variables`

* 开启tensorflow serving服务

    * 配置 `models.config` 文件

    * 配置 `monitor.config` 文件

    * 可使用 `docker ps` 查看正在使用的镜像服务， `docker stop ID` 结束服务

    * 开启服务
	
		```shell
		sudo docker run --rm -p 8502:8501 --mount type=bind,source=/home/wangjian0110/myWork/work/tf_serving_test/release,target=/models   -t tensorflow/serving  --rest_api_num_threads=4 --model_config_file=/models/models.config --monitoring_config_file=/models/monitor.config   
		# source：模型存放的本地目录
		# MODEL_NAME:模型名
		```
		
    * 使用 `curl 0.0.0.0:8502/v1/models/test_model/metadata` 获取server的输入输出节点信息

## 模型2-gunicorn+flask

* 开启flask服务

    `gunicorn -w 4 -b 0.0.0.0:7123 --timeout 600 flask_serving:app`

## 两种模型性能对比

### 耗时对比

* 服务总耗时对比（单位：`ms`）

	耗时= 得到response时的时间 - 发起post时的时间

	小数据：

	```
	python server_api.py --batch_test=true
	```

	大数据：

	```
	python server_api.py --batch_test=true --batch_type=large
	```


	* 小数据情况下，tensorflow serving 为 gunicorn+flask的 `10-20%`

	|序号|batch size|tf-serving|flask|rate(tf-serving/flask)
	| :-: | :-: | :-: | :-: | :-: 
	|1|1|2.1680593490600586|12.59925365447998|0.17207839515868076
	|2|2|2.383112907409668|13.478684425354004|0.1768060466588955
	|3|3|2.6692628860473633|16.274523735046387|0.16401480802164656
	|4|4|2.8803110122680664|18.027424812316895|0.15977384691684576
	|5|5|3.0861377716064453|18.974971771240234|0.16264254876436796
	|6|6|3.285384178161621|20.71101665496826|0.1586297878512645
	|7|7|3.4763097763061523|23.314905166625977|0.14910246262902674
	|8|8|3.765583038330078|24.078750610351562|0.1563861472410133
	|9|9|3.9990186691284175|28.97038459777832|0.1380381629257036
	|10|10|4.373025894165039|33.05943012237549|0.13227771555581838

	* 大数据情况下，tensorflow serving 约为 gunicorn+flask的 `10%`

	|序号|batch size|tf-serving|flask|rate(tf-serving/flask)
	| :-: | :-: | :-: | :-: | :-: 
	|1|1|2.7808189392089844|12.457132339477539|0.22323106662328465
	|2|10|4.418444633483887|38.072800636291504|0.11605252462757273
	|3|100|24.031996726989746|296.23560905456543|0.08112460484979424
	|4|1000|279.4727563858032|2394.981360435486|0.11669099434451796
	|5|10000|2556.168270111084|29140.264105796814|0.0877194613209628

* inference耗时对比

	由于tensorflow serving inference的时间不好拿到，以上表中的服务总时间作为inference时间，真实的时间应该比此时间小。

	flask下inference时间为模型中tf.Session.run花费的时间

	* 小数据情况下

	|序号|batch size|tf-serving|flask|rate(tf-serving/flask)
	| :-: | :-: | :-: | :-: | :-: 
	|1|1|2.1680593490600586|6.559133529663086|0.33054051106829996
	|2|2|2.383112907409668|6.8466901779174805|0.34806787593454774
	|3|3|2.6692628860473633|8.260655403137207|0.3231296738311634
	|4|4|2.8803110122680664|9.04541015625|0.3184279057017544
	|5|5|3.0861377716064453|9.276413917541504|0.33268651000691374
	|6|6|3.285384178161621|10.282254219055176|0.3195198356478207
	|7|7|3.4763097763061523|11.585116386413574|0.3000668841258245
	|8|8|3.765583038330078|11.267971992492676|0.33418462886124584
	|9|9|3.9990186691284175|14.677286148071289|0.27246308539497405
	|10|10|4.373025894165039|17.45467185974121|0.250536127479504

	* 大数据情况下

	|序号|batch size|tf-serving|flask|rate(tf-serving/flask)|
	| :-: | :-: | :-: | :-: | :-: |
	|1|1|2.7808189392089844|5.994391441345215|0.4639034614971582
	|2|10|4.418444633483887|18.738627433776855|0.2357933978408433
	|3|100|24.031996726989746|123.89552593231201|0.19396985118025306
	|4|1000|279.4727563858032|910.8529329299927|0.3068253351139873
	|5|10000|2556.168270111084|7724.274325370789|0.3309266556879284

* 高并发性能测试

	使用python多线程模拟高并发任务，两种服务指定的work数都为4

	```shell
	python  server_api.py --concurrent_test=true
	```

	|序号|并发数|tf-serving|flask|rate(tf-serving/flask)
	| :-: | :-: | :-: | :-: | :-: 
	|1|1|5.176782608032227|18.38850975036621|0.2815226833663957
	|2|10|24.54996109008789|102.15520858764648|0.24032020911615748
	|3|100|236.53340339660645|915.2920246124268|0.2584239751206886
	|4|1000|2332.703113555908|8629.108905792236|0.27032954839521095
	|5|10000|23560.18304824829|85167.12021827698|0.27663472696816915






# tensorflow serving 主要参数


```
Flags:
	--port=8500                      	int32	Port to listen on for gRPC API
	--grpc_socket_path=""            	string	If non-empty, listen to a UNIX socket for gRPC API on the given path. Can be either relative or absolute path.
	--rest_api_port=0                	int32	Port to listen on for HTTP/REST API. If set to zero HTTP/REST API will not be exported. This port must be different than the one specified in --port.
	--rest_api_num_threads=16        	int32	Number of threads for HTTP/REST API processing. If not set, will be auto set based on number of CPUs.
	--rest_api_timeout_in_ms=30000   	int32	Timeout for HTTP/REST API calls.
	--enable_batching=false          	bool	enable batching
	--batching_parameters_file=""    	string	If non-empty, read an ascii BatchingParameters protobuf from the supplied file name and use the contained values instead of the defaults.
	--model_config_file=""           	string	If non-empty, read an ascii ModelServerConfig protobuf from the supplied file name, and serve the models in that file. This config file can be used to specify multiple models to serve and other advanced parameters including non-default version policy. (If used, --model_name, --model_base_path are ignored.)
	--model_name="default"           	string	name of model (ignored if --model_config_file flag is set)
	--model_base_path=""             	string	path to export (ignored if --model_config_file flag is set, otherwise required)
	--max_num_load_retries=5         	int32	maximum number of times it retries loading a model after the first failure, before giving up. If set to 0, a load is attempted only once. Default: 5
	--load_retry_interval_micros=60000000	int64	The interval, in microseconds, between each servable load retry. If set negative, it doesn't wait. Default: 1 minute
	--file_system_poll_wait_seconds=1	int32	Interval in seconds between each poll of the filesystem for new model version. If set to zero poll will be exactly done once and not periodically. Setting this to negative value will disable polling entirely causing ModelServer to indefinitely wait for a new model at startup. Negative values are reserved for testing purposes only.
	--flush_filesystem_caches=true   	bool	If true (the default), filesystem caches will be flushed after the initial load of all servables, and after each subsequent individual servable reload (if the number of load threads is 1). This reduces memory consumption of the model server, at the potential cost of cache misses if model files are accessed after servables are loaded.
	--tensorflow_session_parallelism=0	int64	Number of threads to use for running a Tensorflow session. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
	--tensorflow_intra_op_parallelism=0	int64	Number of threads to use to parallelize the executionof an individual op. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
	--tensorflow_inter_op_parallelism=0	int64	Controls the number of operators that can be executed simultaneously. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
	--ssl_config_file=""             	string	If non-empty, read an ascii SSLConfig protobuf from the supplied file name and set up a secure gRPC channel
	--platform_config_file=""        	string	If non-empty, read an ascii PlatformConfigMap protobuf from the supplied file name, and use that platform config instead of the Tensorflow platform. (If used, --enable_batching is ignored.)
	--per_process_gpu_memory_fraction=0.000000	float	Fraction that each process occupies of the GPU memory space the value is between 0.0 and 1.0 (with 0.0 as the default) If 1.0, the server will allocate all the memory when the server starts, If 0.0, Tensorflow will automatically select a value.
	--saved_model_tags="serve"       	string	Comma-separated set of tags corresponding to the meta graph def to load from SavedModel.
	--grpc_channel_arguments=""      	string	A comma separated list of arguments to be passed to the grpc server. (e.g. grpc.max_connection_age_ms=2000)
	--enable_model_warmup=true       	bool	Enables model warmup, which triggers lazy initializations (such as TF optimizations) at load time, to reduce first request latency.
	--version=false                  	bool	Display version
	--monitoring_config_file=""      	string	If non-empty, read an ascii MonitoringConfig protobuf from the supplied file name
```