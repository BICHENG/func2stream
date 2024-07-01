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
__version__ = "0.1.0"
__license__ = "MPL2.0"


import os, time, threading, inspect, traceback, queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np


from basicnconst import *

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
        assertl(callable(fn), ELEMENT_FN_NOT_CALLABLE.format(friendly_name, fn.__name__))
        assertl(isinstance(kwargs, dict), ELEMENT_KWARGS_NOT_DICT.format(friendly_name, kwargs))
        
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        fn_params = [param.name for param in params[1:]]
        missing_params = [param.name for param in  params[1:] if all([
            param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD],
            param.default == inspect.Parameter.empty,
            param.name not in kwargs])]
        extra_kwargs = set(kwargs.keys()) - set(fn_params)
        
        assertl(params, ELEMENT_PARAMS_REQUIRED.format(friendly_name, fn.__name__))
        assertl(not params[0].default != inspect.Parameter.empty, ELEMENT_FIRST_PARAM_NO_DEFAULT.format(friendly_name, params[0].name))
        assertl(not missing_params, ELEMENT_MISSING_PARAMS.format(friendly_name, len(missing_params), missing_params, fn_params))
        assertl(not extra_kwargs, ELEMENT_EXTRA_KWARGS.format(friendly_name, len(extra_kwargs), extra_kwargs))
        
        
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
            logger.info(ELEMENT_STARTED.format(self.friendly_name))
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
                logger.info(ELEMENT_ERROR_OCCURRED.format(
                    self.friendly_name, # Element Name
                    e,                  # Exception
                    self.fn.__name__,   # Function Name
                    self.kwargs,        # Function Arguments
                    traceback_info))    # Traceback
                time.sleep(1)
        logger.info(ELEMENT_STOPPED.format(self.friendly_name))
        
    def _link_to(self, other,depth=1):
        assertl(isinstance(other, Element), ELEMENT_OTHER_NOT_INSTANCE.format(other))
        if all([self.sink is None, other.source is None]): self.sink = _queue(depth, leaky=False); other.set_source(self.sink)
        if all([self.sink is None, other.source is not None]): self.set_sink(other.source)
        if all([self.sink is not None, other.source is None]): other.set_source(self.sink)
        return other
    
    def __call__(self, item):
        # 首先, 需要确保有source和sink
        assertl(self.thread is not None, ELEMENT_THREAD_NOT_STARTED.format(self.friendly_name))
        self.source.put(item)
        return self.sink.get()
    
    def start(self):
        assertl(self.source is not None, ELEMENT_NO_SOURCE_QUEUE.format(self.friendly_name))
        assertl(self.sink is not None, ELEMENT_NO_SINK_QUEUE.format(self.friendly_name))
        assertl(self.thread is None, ELEMENT_ALREADY_STARTED.format(self.friendly_name))
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
        # execution time of the element(average, max, min, top 5%, bottom 5%)
        exec_times = np.array(self.exec_times)
        t_avg, t_max, t_min, t_95, t_5 = np.mean(exec_times)*1000, np.max(exec_times)*1000, np.min(exec_times)*1000, np.percentile(exec_times, 95)*1000, np.percentile(exec_times, 5)*1000
        if print_summary:
            logger.info("".join([
                ELEMENT_EXEC_TIME_NAME.format(self.friendly_name),
                ELEMENT_EXEC_TIME_AVG.format(t_avg),
                ELEMENT_EXEC_TIME_MAX.format(t_max),
                ELEMENT_EXEC_TIME_MIN.format(t_min),
                ELEMENT_EXEC_TIME_TOP_5.format(t_95),
                ELEMENT_EXEC_TIME_BTN_5.format(t_5)
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
                logger.info(ELEMENT_STARTED.format(self.friendly_name))
                while not self.stop_flag.is_set(): self.sink.put(self.reader_call())
            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                logger.info(DSOURCE_ERROR_OCCURRED.format(
                    self.friendly_name,         # Element Name
                    e,                          # Exception
                    self.reader_call.__name__,  # Function Name
                    self.kwargs,                # Function Arguments
                    traceback_info))            # Traceback
                time.sleep(1)
            logger.info(ELEMENT_STOPPED.format(self.friendly_name))
    def start(self):
        self.source = _queue(1, leaky=False)
        return super().start()

class Pipeline(Element):
    def __init__(self, elements: list, friendly_name="Pipeline",depth=1):
        super().__init__(friendly_name, fn=None, kwargs={}, source=None, sink=None)        
        assertl(len(elements) > 1, PIPELINE_TOO_FEW_ELEMENTS.format(len(elements)))
        self.elements = elements
        # For each item in elements, check if it is an instance of Element and try to convert it to an instance of Element
        for i, elm in enumerate(self.elements):
            if not isinstance(elm, Element):
                if callable(elm):
                    self.elements[i] = Element(elm.__name__, elm)
                if isinstance(elm, tuple) and len(elm) == 2 and callable(elm[0]) and isinstance(elm[1], dict):
                    self.elements[i] = Element(elm[0].__name__, elm[0], elm[1])                    
        # For each pair of adjacent elements in elements, create a connection queue
        logger.info(ELEMENT_CONNECTION_BGN.format(self.friendly_name, len(self.elements)))
        for i in range(len(self.elements) - 1):
            self.elements[i]._link_to(self.elements[i + 1], depth=depth)
            logger.info(ELEMENT_LINK_ESTABLISH.format(self.elements[i].friendly_name, depth, self.elements[i + 1].friendly_name, i + 1, len(self.elements) - 1))
        logger.info(ELEMENT_CONNECTION_END)

        for i, elm in enumerate(self.elements):
            elm.friendly_name = f"{self.friendly_name}/{elm.friendly_name} [{i+1}/{len(self.elements)}]"
            if i > 0 and i < len(self.elements) - 1: elm.start()
         
        # The source and sink of the pipeline itself are delegated to the first and last elements to interact with the outside world,
        # which is actually the interaction with these two specific elements in the pipeline.
        self.source = self.elements[0].source
        self.sink = self.elements[-1].sink
        
        # Don't use if you dk what you are doing
        def _set_source(source):
            logger.info(SET_SRC_QUEUE.format(self.friendly_name))
            self.elements[0].source = source;return self
        def _set_sink(sink):
            logger.info(SET_DST_QUEUE.format(self.friendly_name))
            self.elements[-1].sink = sink;return self
        
        self.set_source=_set_source
        self.set_sink=_set_sink
        
    def start(self):
        assertl(any([self.elements[0].source is not None, isinstance(self.elements[0], DataSource)]), PIPELINE_NO_SOURCE.format(self.elements[0].friendly_name, self.friendly_name))
        if self.elements[-1].sink is None:
            self.elements[-1].sink = _queue(1, leaky=True)
            logger.info(ELEMENT_SINK_IS_LEAKY.format(self.elements[-1].friendly_name))
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
    
    def exec_time_summary(self, print_summary=True):
        exec_times = [element.exec_time_summary(print_summary=False) for element in self.elements]
        msg = [EXEC_TIME_SUMMARY_HEADER.format(self.friendly_name, len(self.elements))]
        for i, (t_avg, t_max, t_min, t_95, t_5) in enumerate(exec_times):
            msg.append(ELEMENT_EXEC_TIME_NAME.format(self.elements[i].friendly_name))
            msg.append(ELEMENT_EXEC_TIME_AVG.format(t_avg))
            msg.append(ELEMENT_EXEC_TIME_MAX.format(t_max))
            msg.append(ELEMENT_EXEC_TIME_MIN.format(t_min))
            msg.append(ELEMENT_EXEC_TIME_TOP_5.format(t_95))
            msg.append(ELEMENT_EXEC_TIME_BTN_5.format(t_5))
        if print_summary:
            logger.info("\n".join(msg))
        return exec_times

    def exec_time_summary_lite(self, print_summary=True):
        exec_times = [element.exec_time_summary(print_summary=False) for element in self.elements]
        msg = [EXEC_TIME_SUMMARY_HEADER.format(self.friendly_name, len(self.elements))]
        for i, (t_avg, t_max, t_min, t_95, t_5) in enumerate(exec_times):
            msg.append(ELEMENT_EXEC_TIME_NAME.format(self.elements[i].friendly_name) + ELEMENT_EXEC_TIME_TOP_5.format(t_95))
        
        most_time_consuming = np.argmax([t[0] for t in exec_times])
        msg.append(MOST_TIME_CONSUMING_HEADER.format(self.elements[most_time_consuming].friendly_name))
        msg.append(ELEMENT_EXEC_TIME_AVG.format(exec_times[most_time_consuming][0]))
        msg.append(ELEMENT_EXEC_TIME_MAX.format(exec_times[most_time_consuming][1]))
        msg.append(ELEMENT_EXEC_TIME_MIN.format(exec_times[most_time_consuming][2]))
        msg.append(ELEMENT_EXEC_TIME_TOP_5.format(exec_times[most_time_consuming][3]))
        msg.append(ELEMENT_EXEC_TIME_BTN_5.format(exec_times[most_time_consuming][4]))
        
        if print_summary:
            logger.info("\n".join(msg))
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
        self.friendly_name = f"{friendly_name if friendly_name else 'MapReduce'}[{'ReadOnly' if nocopy else 'Copy'}]━┓"
        for fn_name, kwargs in zip(fn_names, self.kwargs_list):
            self.friendly_name += f"\n\t┣━{fn_name}({kwargs})"
        self.friendly_name += f"\n\t┗━T{len(self.fn_list)}"
    
    def ctx_mode(self,get=None, ret=None):
        self.get = get or [[] for _ in self.fn_list]  # List of lists by default
        self.ret = ret or [[] for _ in self.fn_list]  # List of lists by default
        def _fn_ctx(item):
            futures = []
            for index, fn in enumerate(self.fn_list):
                # Item will be deconstructed according to the get list
                args = [item[key] for key in self.get[index]] if self.get[index] else [item]
                # Submit tasks with deconstructed parameters as input to fn
                future = self.exec.submit(fn, *args)
                futures.append(future)

            # Wait for all tasks to complete
            wait(futures)

            # Process the return result and fill it back to item according to the ret list
            for index, future in enumerate(futures):
                result = future.result()
                if self.ret[index]:
                    for key, value in zip(self.ret[index], result if isinstance(result, tuple) else [result]):
                        item[key] = value
            return item # All results are stored in item

        self.fn = _fn_ctx
        return self

import functools
def from_ctx(get=None, ret=None):
    if get is None: get = []
    if ret is None: ret = []    
    def decorator(func):
        @functools.wraps(func)  # keep the name and docstring of the original function
        def wrapper(ctx):          
            assertl(isinstance(ctx, dict), FUNC_CTX_NOT_DICT.format(func.__name__, type(ctx).__name__))

            missing_keys = [k for k in get if k not in ctx]
            assertl(not missing_keys, FUNC_MISSING_KEYS.format(func.__name__, missing_keys))
            
            if len(get): result = func([ctx[g] for g in get] if len(get) > 1 else ctx[get[0]])
            else: result = func()
            
            if not ret: return ctx
            if not isinstance(result, tuple): result = (result,)
            
            assertl(len(ret) == len(result), FUNC_RETURN_MISMATCH.format(func.__name__, len(result), ret, len(ret)))
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
        assertl(isinstance(local_vars, dict), CTX_LOCALVARS_NOT_DICT)
        return ContextContainer(local_vars)
    return wrapper