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


import os, time, threading, inspect, traceback, queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
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
        assert callable(fn), f"[{friendly_name}] 元素无法创建, 函数 {fn.__name__} 不是可调用的"
        assert isinstance(kwargs, dict), f"[{friendly_name}] 元素无法创建, 参数 {kwargs} 不是字典类型"
        
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        fn_params = [param.name for param in params[1:]]
        missing_params = [param.name for param in  params[1:] if all([
            param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD],
            param.default == inspect.Parameter.empty,
            param.name not in kwargs])]
        extra_kwargs = set(kwargs.keys()) - set(fn_params)
        
        assert params, f"{friendly_name}: 处理函数需要至少一个位置参数, 例如: def {fn.__name__}(item, ...)"
        assert not params[0].default != inspect.Parameter.empty, f"{friendly_name}: 第一个位置参数 {params[0].name} 不能有默认值"
        assert not missing_params, f"{friendly_name}: 缺少{len(missing_params)}个必需参数：{missing_params}，有效参数有：{fn_params}"
        assert not extra_kwargs, f"{friendly_name}: 提供了{len(extra_kwargs)}个多余参数：{extra_kwargs}"
        
        
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
            print(f"已启动 {self.friendly_name}")
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
                print(f"{self.friendly_name} 元素发生异常：",
                    f"\t{e} 位于 {self.fn.__name__}, 参数：{self.kwargs}",
                    f"\ttraceback: {traceback_info}")
                time.sleep(1)
        print(f"{self.friendly_name} 已停止")
        
    def _link_to(self, other,depth=1):
        assert isinstance(other, Element), f"{other} 不是 Element 类的实例"
        if all([self.sink is None, other.source is None]): self.sink = _queue(depth, leaky=False); other.set_source(self.sink)
        if all([self.sink is None, other.source is not None]): self.set_sink(other.source)
        if all([self.sink is not None, other.source is None]): other.set_source(self.sink)
        return other
    
    def __call__(self, item):
        # 首先, 需要确保有source和sink
        assert self.thread is not None, f"{self.friendly_name} 元素没有启动, 无法处理元素"
        self.source.put(item)
        return self.sink.get()
    
    def start(self):
        assert self.source is not None, f"{self.friendly_name} 元素没有设置输入队列, 无法启动"
        assert self.sink is not None, f"{self.friendly_name} 元素没有设置输出队列, 无法启动"
        assert self.thread is None, f"{self.friendly_name} 元素已经启动, 请勿重复启动"
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
                f"{self.friendly_name} 执行时间统计：",
                f"\t平均处理时间：{t_avg:.2f} ms",
                f"\t最大处理时间：{t_max:.2f} ms",
                f"\t最小处理时间：{t_min:.2f} ms",
                f"\ttop 5% 处理时间：{t_95:.2f} ms",
                f"\tbottom 5% 处理时间：{t_5:.2f} ms"
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
                print(f"已启动 {self.friendly_name} ")
                while not self.stop_flag.is_set(): self.sink.put(self.reader_call())
            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                print(f"{self.friendly_name} 源发生异常：",
                    f"\t{e} 位于 {self.reader_call.__name__}, 参数：{self.kwargs}",
                    f"\ttraceback: {traceback_info}")
                time.sleep(1)
            print(f"{self.friendly_name} 已停止")
    def start(self):
        self.source = _queue(1, leaky=False)
        return super().start()

class Pipeline(Element):
    def __init__(self, elements: list, friendly_name="Pipeline",depth=1):
        super().__init__(friendly_name, fn=None, kwargs={}, source=None, sink=None)        
        assert len(elements) > 1, f"Pipeline 至少需要2个元素, 目前只有 {len(elements)} 个"        
        self.elements = elements
        # 针对elements中每个item, 检查是否是Element的实例, 并尝试转换为Element的实例
        for i, elm in enumerate(self.elements):
            if not isinstance(elm, Element):
                if callable(elm):
                    self.elements[i] = Element(elm.__name__, elm)
                if isinstance(elm, tuple) and len(elm) == 2 and callable(elm[0]) and isinstance(elm[1], dict):
                    self.elements[i] = Element(elm[0].__name__, elm[0], elm[1])                    
        # 针对elements中每对相邻元素, 创建连接队列
        print(f"构建\t┌{self.friendly_name} 的连接, 共 {len(self.elements)} 个元素：┐")
        for i in range(len(self.elements) - 1):
            self.elements[i]._link_to(self.elements[i + 1],depth=depth)
            print(f"\t│连接 {self.elements[i].friendly_name} -{depth}--> {self.elements[i + 1].friendly_name} 已建立 [{i+1}/{len(self.elements)-1}]")
        print("\t└───────────────────")
        for i, elm in enumerate(self.elements):
            elm.friendly_name = f"{self.friendly_name}/{elm.friendly_name} [{i+1}/{len(self.elements)}]"
            if i > 0 and i < len(self.elements) - 1: elm.start()
         
        # Pipeline 自身的源头（source）和汇点（sink）委托给了首个和末尾元素以实现与外部世界的接口, 实际上是与 Pipeline 中的这两个特定元素的交互。
        self.source = self.elements[0].source
        self.sink = self.elements[-1].sink
        
        def _set_source(source):
            print(f"设置 {self.friendly_name} 的输入队列")
            self.elements[0].source = source;return self
        def _set_sink(sink):
            print(f"设置 {self.friendly_name} 的输出队列")
            self.elements[-1].sink = sink;return self
        
        self.set_source=_set_source
        self.set_sink=_set_sink
        
    def start(self):
        assert any([
            self.elements[0].source is not None,
            isinstance(self.elements[0], DataSource)
            ]), f"{self.elements[0].friendly_name}@{self.friendly_name} 没有设置输入队列, 无法启动"        
        if self.elements[-1].sink is None:
            self.elements[-1].sink = _queue(1, leaky=True)
            print(f"SINK {self.elements[-1].friendly_name} 将是一个自动丢弃数据的队列")
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
    
    # def set_input(self,q:Queue):
    #     self.element[0].source = q
    #     return self
    
    def exec_time_summary(self,print_summary=True):
        exec_times = [element.exec_time_summary(print_summary=False) for element in self.elements]
        msg = [f"{self.friendly_name} 共 {len(self.elements)} 个元素, 执行时间统计：",]
        for i, (t_avg, t_max, t_min, t_95, t_5) in enumerate(exec_times):
            msg.append(f"\t{self.elements[i].friendly_name}：")
            msg.append(f"\t\t平均处理时间：{t_avg:.2f} ms")
            msg.append(f"\t\t最大处理时间：{t_max:.2f} ms")
            msg.append(f"\t\t最小处理时间：{t_min:.2f} ms")
            msg.append(f"\t\ttop 5% 处理时间：{t_95:.2f} ms")
            msg.append(f"\t\tbottom 5% 处理时间：{t_5:.2f} ms")
        if print_summary: print("\n".join(msg))
        return exec_times
    
    def exec_time_summary_lite(self,print_summary=True):
        exec_times = [element.exec_time_summary(print_summary=False) for element in self.elements]
        msg = [f"{self.friendly_name} 共 {len(self.elements)} 个元素, 执行时间统计：",]
        for i, (t_avg, t_max, t_min, t_95, t_5) in enumerate(exec_times):
            msg.append(f"\t{self.elements[i].friendly_name}：top 5% 处理时间：{t_95:.2f} ms")
            
        most_time_consuming = np.argmax([t[0] for t in exec_times])
        msg.append(f"\n\t最耗时的元素是 {self.elements[most_time_consuming].friendly_name}：")
        msg.append(f"\t\平均处理时间：{exec_times[most_time_consuming][0]:.2f} ms")
        msg.append(f"\t\最大处理时间：{exec_times[most_time_consuming][1]:.2f} ms")
        msg.append(f"\t\最小处理时间：{exec_times[most_time_consuming][2]:.2f} ms")
        msg.append(f"\ttop 5% 处理时间：{exec_times[most_time_consuming][3]:.2f} ms")
        msg.append(f"\tbottom 5% 处理时间：{exec_times[most_time_consuming][4]:.2f} ms")
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
            assert isinstance(ctx, dict), f"{func.__name__}需要接收一个字典类型的上下文，但是接收到了 {type(ctx).__name__} 类型，请检查上游的输出。"

            missing_keys = [k for k in get if k not in ctx]
            assert not missing_keys, f"{func.__name__}需要获取的键: {missing_keys} 不在上下文中，请检查上游的输出。"
            
            if len(get): result = func([ctx[g] for g in get] if len(get) > 1 else ctx[get[0]])
            else: result = func()
            
            if not ret: return ctx
            if not isinstance(result, tuple): result = (result,)
            
            assert len(ret) == len(result), f"{func.__name__}返回的结果数量: {len(result)} 与设置的键({ret})数量: {len(ret)} 不匹配，请检查函数的返回值。"
            for key, value in zip(ret, result): ctx[key] = value
             
            return ctx
        wrapper.fn = func
        wrapper.get = get
        wrapper.ret = ret
        return wrapper
    return decorator

def build_ctx(key,constants={"_":None},init_dict={}):
       
    def ctx_fn(x): 
        d = {k: v for k, v in init_dict.items()}
        for k, v in constants.items(): d[k] = v
        d[key] = x  
        return d
    return ctx_fn

class ContextContainer(dict):
    def __init__(self, local_vars):
        super().__init__()
        for name, var in local_vars.items():
            self[name] = var
            setattr(self, name, var)
            
def init_ctx(func):
    def wrapper(*args, **kwargs):
        local_vars = func(*args, **kwargs)
        return ContextContainer(local_vars)
    return wrapper