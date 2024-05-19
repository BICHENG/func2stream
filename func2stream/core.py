"""
core.py

Effortlessly transform functions into asynchronous elements for building high-performance pipelines.

Author: BI CHENG
GitHub: https://github.com/BICHENG/func2stream
License: MPL2.0
Created: 2024/5/1

For Usage, please refer to https://github.com/BICHENG/func2stream/samples or README.md
"""

__author__ = "BI CHENG"
__version__ = "0.0.0"
__license__ = "MPL2.0"


import time,threading,inspect,traceback,queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor,wait
import numpy as np

class _queue:
    def __init__(self,depth,leaky=False):
        self.depth = depth
        self.queue = queue.Queue(depth)
        self.leaky = leaky
    def put(self,item):
        if self.queue.full() and self.leaky:
            self.queue.get()
        self.queue.put(item)

    
    def get(self):      return self.queue.get()    
    def qsize(self):    return self.queue.qsize()
    def empty(self):    return self.queue.empty()
    def full(self):     return self.queue.full()
    def clear(self):
        while not self.queue.empty():
            self.queue.get()
            
class Element:
    def __init__(self, friendly_name, fn, kwargs={}, source=None, sink=None):
        if fn is None:
            fn = lambda x: x
            kwargs = {}

        assert callable(fn), f"Element {friendly_name} cannot be created, {fn.__name__} is not callable"
        assert isinstance(kwargs, dict), f"Element {friendly_name} cannot be c。/reated, {kwargs} is not a dictionary"
        
        
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        fn_params = [param.name for param in params[1:]]
        missing_params = [param.name for param in  params[1:] if all([
            param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD],
            param.default == inspect.Parameter.empty,
            param.name not in kwargs])]
        extra_kwargs = set(kwargs.keys()) - set(fn_params)
        
        assert params, f"{friendly_name}: The processing function needs at least one positional parameter, e.g. def {fn.__name__}(item, ...)"
        assert not params[0].default != inspect.Parameter.empty, f"{friendly_name}: The first positional parameter {params[0].name} cannot have a default value"
        assert not missing_params, f"{friendly_name}: Missing {len(missing_params)} required parameters: {missing_params}, valid parameters are: {fn_params}"
        assert not extra_kwargs, f"{friendly_name}: Provided {len(extra_kwargs)} extra parameters: {extra_kwargs}"
        
                
        
        self.friendly_name = friendly_name
        self.fn = fn
        self.kwargs = kwargs
        self.source = source
        self.sink = sink
        
        self.cnt = 0
        self.thread = None
        self.stop_flag = threading.Event()
        
        self.exec_times = deque(maxlen=50)
        self.exec_times.append(0)
    
    def set_source(self, source: _queue):
        self.source = source
        return self
    def set_sink(self, sink: _queue):
        self.sink = sink
        return self
    def get_source(self):
        return self.source
    def get_sink(self):
        return self.sink
    
    def _worker(self):        
        while not self.stop_flag.is_set():
            print(f"{self.friendly_name} has started")
            try:
                while not self.stop_flag.is_set():
                    if self.source.empty():
                        time.sleep(0.0001)
                        continue
                    item = self.source.get()
                    t0 = time.time()
                    result = self.fn(item, **self.kwargs)
                    self.exec_times.append(time.time()-t0)
                    self.sink.put(result)
                    self.cnt += 1

            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                print(f"{self.friendly_name} element has encountered an exception：",
                    f"\t{e} occurred in {self.fn.__name__}, with arguments：{self.kwargs}",
                    f"\ttraceback: {traceback_info}")
                
                time.sleep(1)
        print(f"{self.friendly_name} has stopped")
        
    def _link_to(self, other):
        assert isinstance(other, Element), f"{other} is not an instance of Element"
        
        if all([self.sink is None, other.source is None]): self.sink = _queue(1, leaky=False); other.set_source(self.sink)
        if all([self.sink is None, other.source is not None]): self.set_sink(other.source)
        if all([self.sink is not None, other.source is None]): other.set_source(self.sink)
        return other
    
    def __call__(self, item):
        assert self.source is not None, f"{self.friendly_name} element has no input queue, cannot process item"        
        self.source.put(item)
        return self.sink.get()
    
    def start(self):
        assert self.source is not None, f"{self.friendly_name} element has no input queue, cannot start"
        assert self.sink is not None, f"{self.friendly_name} element has no output queue, cannot start"
        assert self.thread is None, f"{self.friendly_name} element has already started, cannot start again"
        
        self.thread = threading.Thread(target=self._worker, name=self.friendly_name, daemon=True)
        self.thread.start()
        return self
    
    def stop(self):
        self.stop_flag.set()
        if self.thread is not None: self.thread.join()
        return self
    
    def time_per_item(self):
        return np.mean(self.exec_times) if len(self.exec_times) > 0 else 0
    
    def exec_time_summary(self,print_summary=True):
        # 最近执行的平均时间、最大时间、最小时间、top 5% 和 bottom 5%
        exec_times = np.array(self.exec_times)
        t_avg, t_max, t_min, t_95, t_5 = np.mean(exec_times)*1000, np.max(exec_times)*1000, np.min(exec_times)*1000, np.percentile(exec_times, 95)*1000, np.percentile(exec_times, 5)*1000
        if print_summary:
            print("".join([
                f"{self.friendly_name} Execution Time Summary：",
                f"\tAverage Processing Time：{t_avg:.2f} ms",
                f"\tMaximum Processing Time：{t_max:.2f} ms",
                f"\tMinimum Processing Time：{t_min:.2f} ms",
                f"\tTop 5% Processing Time：{t_95:.2f} ms",
                f"\tBottom 5% Processing Time：{t_5:.2f} ms"
                ]))
            
            
            
        return t_avg, t_max, t_min, t_95, t_5
   
class DataSource(Element):
    def __init__(self, reader_call,
                 friendly_name=""
                 ):
        super().__init__(reader_call.__name__ if friendly_name == "" else friendly_name, fn=None, kwargs={}, source=None, sink=None)
        self.reader_call = reader_call
    
    def _worker(self):
        while not self.stop_flag.is_set():    
            try:
                print(f"{self.friendly_name} has started")
                while not self.stop_flag.is_set(): self.sink.put(self.reader_call())
            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                print(f"{self.friendly_name} source has encountered an exception：",
                    f"\t{e} occurred in {self.reader_call.__name__}, with arguments：{self.kwargs}",
                    f"\ttraceback: {traceback_info}")                
                time.sleep(1)
            print(f"{self.friendly_name} has stopped")
    def start(self):
        self.source = _queue(1, leaky=False)
        return super().start()

class Pipeline(Element):
    def __init__(self, elements: list, friendly_name="Pipeline"):
        super().__init__(friendly_name, fn=None, kwargs={}, source=None, sink=None)
        assert len(elements) > 1, f"Pipeline needs at least 2 elements, but only {len(elements)} found"  
        self.elements = elements
        # 针对elements中每个item, 检查是否是Element的实例, 并尝试转换为Element的实例
        for i, elm in enumerate(self.elements):
            if not isinstance(elm, Element):
                if callable(elm):
                    self.elements[i] = Element(elm.__name__, elm)
                if isinstance(elm, tuple) and len(elm) == 2 and callable(elm[0]) and isinstance(elm[1], dict):
                    self.elements[i] = Element(elm[0].__name__, elm[0], elm[1])                    
        # 针对elements中每对相邻元素, 创建连接队列
        print(f"Building\t┌{self.friendly_name} with {len(self.elements)} elements：┐")
        
        for i in range(len(self.elements) - 1):
            self.elements[i]._link_to(self.elements[i + 1])
            print(f"\t│Linked {self.elements[i].friendly_name} -> {self.elements[i + 1].friendly_name} [{i+1}/{len(self.elements)-1}]")
        print("\t└───────────────────")
        for i, elm in enumerate(self.elements):
            elm.friendly_name = f"{self.friendly_name}/{elm.friendly_name} [{i+1}/{len(self.elements)}]"
            if i > 0 and i < len(self.elements) - 1: elm.start()
         
        # Pipeline 自身的源头（source）和汇点（sink）委托给了首个和末尾元素以实现与外部世界的接口, 实际上是与 Pipeline 中的这两个特定元素的交互。
        self.source = self.elements[0].source
        self.sink = self.elements[-1].sink
        
        def _set_source(source):
            print(f"Setting the input queue of {self.friendly_name}")
            self.elements[0].source = source;return self
        def _set_sink(sink):
            print(f"Setting the output queue of {self.friendly_name}")
            self.elements[-1].sink = sink;return self
        
        self.set_source=_set_source
        self.set_sink=_set_sink
        
    def start(self):
        assert any([
            self.elements[-1].sink is not None,
            isinstance(self.elements[-1], DataSource)
            ]), f"{self.elements[-1].friendly_name}@{self.friendly_name} has no output queue, cannot start"
        
        if self.elements[-1].sink is None:
            self.elements[-1].sink = _queue(1, leaky=True)
            print(f"SINK {self.elements[-1].friendly_name} will be a pipe that automatically discards old items when full")
        for i in [0, -1]: self.elements[i].start()
        self.source = self.elements[0].source
        self.sink = self.elements[-1].sink
        return self
    
    def stop(self):
        for element in self.elements: element.stop()
        return self
    
    def nodrop(self):
        self.elements[-1].sink = _queue(1, leaky=False)
        return self
    
    def exec_time_summary(self,print_summary=True):
        exec_times = [element.exec_time_summary(print_summary=False) for element in self.elements]
        msg = [f"{self.friendly_name} has {len(self.elements)} elements, execution time summary：",]
        for i, (t_avg, t_max, t_min, t_95, t_5) in enumerate(exec_times):
            msg.append(f"\t{self.elements[i].friendly_name}：")
            msg.append(f"\t\tAverage Processing Time：{t_avg:.2f} ms")
            msg.append(f"\t\tMaximum Processing Time：{t_max:.2f} ms")
            msg.append(f"\t\tMinimum Processing Time：{t_min:.2f} ms")
            msg.append(f"\t\tTop 5% Processing Time：{t_95:.2f} ms")
            msg.append(f"\t\tBottom 5% Processing Time：{t_5:.2f} ms")
        
        if print_summary: print("\n".join(msg))
        return exec_times
    
    def exec_time_summary_lite(self,print_summary=True):
        exec_times = [element.exec_time_summary(print_summary=False) for element in self.elements]
        msg = [f"{self.friendly_name} has {len(self.elements)} elements, execution time summary：",]
        
        for i, (t_avg, t_max, t_min, t_95, t_5) in enumerate(exec_times):
            msg.append(f"\t{self.elements[i].friendly_name}：top 5% processing time：{t_95:.2f} ms")
        
        most_time_consuming = np.argmax([t[0] for t in exec_times])
        msg.append(f"\n\tThe most time-consuming element is {self.elements[most_time_consuming].friendly_name}：")
        msg.append(f"\t\Average Processing Time：{exec_times[most_time_consuming][0]:.2f} ms")
        msg.append(f"\t\Maximum Processing Time：{exec_times[most_time_consuming][1]:.2f} ms")
        msg.append(f"\t\Minimum Processing Time：{exec_times[most_time_consuming][2]:.2f} ms")
        msg.append(f"\t\Top 5% Processing Time：{exec_times[most_time_consuming][3]:.2f} ms")
        msg.append(f"\t\Bottom 5% Processing Time：{exec_times[most_time_consuming][4]:.2f} ms")
        
        
        if print_summary: print("\n".join(msg))
        return exec_times
    
class MapReduce(Element):
    def __init__(self, fn_with_kwargs, friendly_name="", source=None, sink=None,nocopy=True):
        super().__init__(friendly_name, fn=None, kwargs={}, source=source, sink=sink)
        self.fn_list = []
        self.kwargs_list = []
        for v in fn_with_kwargs:
            fn, kwargs = v if isinstance(v, tuple) else (v, {})
            self.fn_list.append(fn)
            self.kwargs_list.append(kwargs)
            
        self.exec = ThreadPoolExecutor(max_workers=len(self.fn_list))
        def _fn_readonly(item):
            return list(self.exec.map(lambda fn, kwargs: fn(item, **kwargs), self.fn_list, self.kwargs_list))
        def _fn_copied_items(item):
            items = [item]+[item.copy() for _ in range(len(self.fn_list)-1)] if hasattr(item, "copy") else [item for _ in range(len(self.fn_list))]
            return list(self.exec.map(lambda item, fn, kwargs: fn(item, **kwargs), items, self.fn_list, self.kwargs_list))
        
        self.fn = _fn_readonly if nocopy else _fn_copied_items
        fn_names = [fn.__name__ for fn in self.fn_list]
        self.friendly_name = f"{friendly_name if friendly_name else 'MapReduce'}[{'ReadOnly' if nocopy else 'Copied'}]━┓"
        for fn_name, kwargs in zip(fn_names, self.kwargs_list):
            self.friendly_name += f"\n\t┣━{fn_name}({kwargs})"
        self.friendly_name += f"\n\t┗━T{len(self.fn_list)}"
    

    def ctx_mode(self,get=None, ret=None):
        self.get = get or [[] for _ in self.fn_list]  # 默认为空列表的列表
        self.ret = ret or [[] for _ in self.fn_list]  # 默认为空列表的列表
        def _fn_ctx(item):
            futures = []
            for index, fn in enumerate(self.fn_list):
                # 根据get列表解构item
                args = [item[key] for key in self.get[index]] if self.get[index] else [item]
                # 提交任务时，将解构的参数作为fn的输入
                future = self.exec.submit(fn, *args)
                futures.append(future)

            # 等待所有任务完成
            wait(futures)

            # 处理返回结果，根据ret列表回填到item
            for index, future in enumerate(futures):
                result = future.result()
                if self.ret[index]:
                    for key, value in zip(self.ret[index], result if isinstance(result, tuple) else [result]):
                        item[key] = value

            return item

        self.fn = _fn_ctx
        return self

import functools
def from_ctx(get=None, ret=None):
    if get is None:
        get = []
    if ret is None:
        ret = []    
    def decorator(func):
        @functools.wraps(func)  # 保持原函数的名字和文档字符串
        def wrapper(ctx):          
            assert isinstance(ctx, dict), f"{func.__name__} expects a dictionary type context, but received a {type(ctx).__name__}, please check the output of the upstream."
            
            missing_keys = [k for k in get if k not in ctx]
            assert not missing_keys, f"{func.__name__} requires keys: {missing_keys} that are not in the context, please check the output of the upstream."

            if len(get): result = func([ctx[g] for g in get] if len(get) > 1 else ctx[get[0]])
            else: result = func()
            
            if not ret: return ctx
            if not isinstance(result, tuple): result = (result,)
            
            assert len(ret) == len(result), f"The number of results returned by {func.__name__}: {len(result)} does not match the number of keys set ({ret}), please check the return value of the function."
                        
            for key, value in zip(ret, result): ctx[key] = value
             
            return ctx
        wrapper.fn = func
        wrapper.get = get
        wrapper.ret = ret
        return wrapper
    return decorator

def build_ctx(key,constants={}):
    def ctx_fn(x):
        d = {key: x}
        for k, v in constants.items(): d[k] = v
        return d
    return ctx_fn

