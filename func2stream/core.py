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
__version__ = "1.0.0-pre"
__license__ = "MPL2.0"


import os, time, threading, inspect, traceback, queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np


from .basicnconst import *

class _queue:
    """High-performance queue using SimpleQueue (Python 3.7+)"""
    __slots__ = ('depth', 'queue', 'leaky', '_size', '_lock')
    
    def __init__(self, depth, leaky=False):
        self.depth = depth
        self.queue = queue.SimpleQueue()
        self.leaky = leaky
        self._size = 0
        self._lock = threading.Lock() if leaky else None
    
    def put(self, item):
        if self.leaky:
            with self._lock:
                while self._size >= self.depth:
                    try:
                        self.queue.get_nowait()
                        self._size -= 1
                    except:
                        break
                self.queue.put(item)
                self._size += 1
        else:
            self.queue.put(item)
    
    def get(self):
        item = self.queue.get()
        if self.leaky:
            with self._lock:
                self._size -= 1
        return item
    
    def get_timeout(self, t):
        item = self.queue.get(timeout=t)
        if self.leaky:
            with self._lock:
                self._size -= 1
        return item
    
    def qsize(self):    return self._size if self.leaky else self.queue.qsize()
    def empty(self):    return self.queue.empty()
    def full(self):     return self._size >= self.depth if self.leaky else False
    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                if self.leaky:
                    with self._lock:
                        self._size -= 1
            except:
                break
            
class Element:
    def __init__(self, friendly_name, fn, kwargs={}, source=None, sink=None, timing=True):
        if fn is None:
            fn = lambda x: x
            kwargs = {}
        assertl(callable(fn), ELEMENT_FN_NOT_CALLABLE.format(friendly_name, fn.__name__))
        assertl(isinstance(kwargs, dict), ELEMENT_KWARGS_NOT_DICT.format(friendly_name, kwargs))
        
        if not getattr(fn, '_auto_ctx', False):
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())

            fn_params = [param.name for param in params[1:]]
            missing_params = [param.name for param in  params[1:] if all([
                param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD],
                param.default == inspect.Parameter.empty,
                param.name not in kwargs])]
            extra_kwargs = set(kwargs.keys()) - set(fn_params)
            
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
        
        self.timing = timing
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
                if self.timing:
                    while not self.stop_flag.is_set():
                        try:
                            item = self.source.get_timeout(0.01)
                        except queue.Empty:
                            continue
                        t0 = time.time()
                        result = self.fn(item, **self.kwargs)
                        self.exec_times.append(time.time()-t0)
                        self.sink.put(result)
                        self.cnt += 1
                else:
                    fn = self.fn
                    kwargs = self.kwargs
                    source_get = self.source.get_timeout
                    sink_put = self.sink.put
                    while not self.stop_flag.is_set():
                        try:
                            item = source_get(0.01)
                        except queue.Empty:
                            continue
                        sink_put(fn(item, **kwargs))
                        self.cnt += 1

            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                logger.info(ELEMENT_ERROR_OCCURRED.format(
                    self.friendly_name, 
                    e, 
                    self.fn.__name__, 
                    self.kwargs, 
                    traceback_info))
                time.sleep(1)
        logger.info(ELEMENT_STOPPED.format(self.friendly_name))
        
    def _link_to(self, other,depth=1):
        assertl(isinstance(other, Element), ELEMENT_OTHER_NOT_INSTANCE.format(other))
        if all([self.sink is None, other.source is None]): self.sink = _queue(depth, leaky=False); other.set_source(self.sink)
        if all([self.sink is None, other.source is not None]): self.set_sink(other.source)
        if all([self.sink is not None, other.source is None]): other.set_source(self.sink)
        return other
    
    def __call__(self, item):
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
    """数据源，自动将读取的数据包装为 ctx dict"""
    def __init__(self, reader_call, friendly_name="", ctx_key="frame"):
        super().__init__(reader_call.__name__ if friendly_name == "" else friendly_name, fn=None, kwargs={}, source=None, sink=None)
        self.reader_call = reader_call
        self.ctx_key = ctx_key
    
    def _worker(self):
        while not self.stop_flag.is_set():    
            try:
                logger.info(ELEMENT_STARTED.format(self.friendly_name))
                if self.ctx_key:
                    while not self.stop_flag.is_set():
                        data = self.reader_call()
                        self.sink.put({"_": None, self.ctx_key: data})
                else:
                    while not self.stop_flag.is_set():
                        self.sink.put(self.reader_call())
            except Exception as e:
                traceback_info = '\t'.join(traceback.format_exception(None, e, e.__traceback__))
                logger.info(DSOURCE_ERROR_OCCURRED.format(
                    self.friendly_name, 
                    e, 
                    self.reader_call.__name__, 
                    self.kwargs, 
                    traceback_info))
                time.sleep(1)
            logger.info(ELEMENT_STOPPED.format(self.friendly_name))
    def start(self):
        self.source = _queue(1, leaky=False)
        return super().start()

class Pipeline(Element):
    """elements 串联成流水线，auto_ctx=True 时自动包装带字符串注解的函数"""
    def __init__(self, elements: list, friendly_name="Pipeline", depth=1, auto_ctx=True):
        super().__init__(friendly_name, fn=None, kwargs={}, source=None, sink=None)        
        assertl(len(elements) > 1, PIPELINE_TOO_FEW_ELEMENTS.format(len(elements)))
        self.elements = elements
        
        if auto_ctx:
            from .implicit_ctx import auto_ctx as _auto_ctx, _should_wrap
            import inspect
            for i, elm in enumerate(self.elements):
                if not isinstance(elm, Element) and callable(elm):
                    if not getattr(elm, '_auto_ctx', False) and getattr(elm, '__name__', '') != 'ctx_fn':
                        if _should_wrap(elm):
                            self.elements[i] = _auto_ctx(elm)
                        else:
                            try:
                                sig = inspect.signature(elm)
                                if sig.return_annotation is inspect.Signature.empty:
                                    logger.warning(
                                            f"⚠️ 函数 {elm.__name__}() 无返回注解，将不被包装。\n"
                                            f"   如需参与数据流，请添加注解：\n"
                                            f"   - 有输出: def {elm.__name__}(...) -> \"output_key\":\n"
                                            f"   - 纯副作用: def {elm.__name__}(...) -> None:"
                                        )
                            except:
                                pass
        
        # 检查 ret=[] 但有 return 的函数（可能忘了注解）
        for i, elm in enumerate(self.elements):
            fn = elm.fn if isinstance(elm, Element) else elm
            if fn is not None and hasattr(fn, 'ret') and fn.ret == []:
                import ast, inspect
                try:
                    source = inspect.getsource(fn.fn if hasattr(fn, 'fn') else fn)
                    tree = ast.parse(source)
                    if any(isinstance(node, ast.Return) and node.value is not None for node in ast.walk(tree)):
                        logger.warning(
                            f"⚠️ 函数 {fn.__name__}() 有返回值但未声明输出键 (ret=[])。\n"
                            f"   返回值将被丢弃！如需保存到 ctx，请添加返回注解：\n"
                            f"   def {fn.__name__}(...) -> \"output_key\":"
                        )
                except:
                    pass
        
        for i, elm in enumerate(self.elements):
            if not isinstance(elm, Element):
                if callable(elm):
                    self.elements[i] = Element(elm.__name__, elm)
                if isinstance(elm, tuple) and len(elm) == 2 and callable(elm[0]) and isinstance(elm[1], dict):
                    self.elements[i] = Element(elm[0].__name__, elm[0], elm[1])
        
        logger.info(ELEMENT_CONNECTION_BGN.format(self.friendly_name, len(self.elements)))
        for i in range(len(self.elements) - 1):
            self.elements[i]._link_to(self.elements[i + 1], depth=depth)
            logger.info(ELEMENT_LINK_ESTABLISH.format(self.elements[i].friendly_name, depth, self.elements[i + 1].friendly_name, i + 1, len(self.elements) - 1))
        logger.info(ELEMENT_CONNECTION_END)

        for i, elm in enumerate(self.elements):
            elm.friendly_name = f"{self.friendly_name}/{elm.friendly_name} [{i+1}/{len(self.elements)}]"
            if i > 0 and i < len(self.elements) - 1: elm.start()
        
        self.source = self.elements[0].source
        self.sink = self.elements[-1].sink
        
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

class ContextContainer(dict):
    def __init__(self, local_vars, auto_wrap_functions=False):
        super().__init__()
        from .implicit_ctx import _should_wrap
        for name, var in local_vars.items():
            if auto_wrap_functions and callable(var) and not name.startswith('_'):
                if not getattr(var, '_auto_ctx', False) and not getattr(var, '_skip_auto_ctx', False):
                    if _should_wrap(var):
                        from .implicit_ctx import auto_ctx
                        var = auto_ctx(var)
            self[name] = var
            setattr(self, name, var)
            
def init_ctx(func=None, *, auto_wrap=True):
    """创建隔离上下文容器，函数需 return locals()，auto_wrap=True 时自动包装内部函数"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            local_vars = fn(*args, **kwargs)
            assertl(isinstance(local_vars, dict), CTX_LOCALVARS_NOT_DICT)
            return ContextContainer(local_vars, auto_wrap_functions=auto_wrap)
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator