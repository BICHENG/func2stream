"""implicit_ctx.py - 基于注解的隐式 Context

参数名 → ctx 读取键，返回注解 → ctx 写入键
字符串注解才包装，类型注解不包装
"""

import inspect
import functools
from typing import List, Callable, Any, Optional

from .basicnconst import assertl, logger


def _should_wrap(func) -> bool:
    """字符串返回注解才包装，避免辅助函数被意外包装"""
    try:
        sig = inspect.signature(func)
        ret_anno = sig.return_annotation
        
        # 只有字符串注解才包装
        if isinstance(ret_anno, str):
            return True
        if isinstance(ret_anno, tuple) and all(isinstance(k, str) for k in ret_anno):
            return True
        
        # 无注解或类型注解都不包装
        return False
    except (ValueError, TypeError):
        return False


def _parse_annotation(func) -> tuple:
    """解析函数注解 → (get_keys, ret_keys) 或 (None, None)"""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None, None
    
    get_keys = []
    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue
        anno = param.annotation
        if anno is inspect.Parameter.empty:
            get_keys.append(name)
        elif isinstance(anno, str):
            get_keys.append(anno)
        else:
            get_keys.append(name)
    
    ret_anno = sig.return_annotation
    
    if ret_anno is inspect.Signature.empty:
        ret_keys = []
    elif isinstance(ret_anno, str):
        ret_keys = [ret_anno]
    elif isinstance(ret_anno, tuple):
        if all(isinstance(k, str) for k in ret_anno):
            ret_keys = list(ret_anno)
        else:
            return None, None
    else:
        return None, None
    
    return get_keys, ret_keys


def _build_ctx_wrapper(func, get_keys: List[str], ret_keys: List[str]):
    """根据 get/ret 数量生成特化 wrapper"""
    n_get = len(get_keys)
    n_ret = len(ret_keys)
    
    if n_get == 0:
        def call_fn(ctx):
            return func()
    elif n_get == 1:
        _key = get_keys[0]
        def call_fn(ctx):
            if _key not in ctx:
                raise KeyError(
                    f"函数 {func.__name__}() 需要参数 '{_key}'，但 ctx 中不存在该键。\n"
                    f"可用的键有: {list(ctx.keys())}\n"
                    f"提示: 检查上游函数是否正确输出了 '{_key}'"
                )
            return func(ctx[_key])
    else:
        def call_fn(ctx):
            missing = [k for k in get_keys if k not in ctx]
            if missing:
                raise KeyError(
                    f"函数 {func.__name__}() 缺少参数: {missing}\n"
                    f"可用的键有: {list(ctx.keys())}\n"
                    f"提示: 检查上游函数是否正确输出了这些键"
                )
            args = [ctx[k] for k in get_keys]
            return func(*args)
    
    _fn_name = func.__name__
    _err_msg = (
        f"管道函数 {_fn_name}() 期望接收 ctx dict，但收到了 {{actual_type}}。\n"
        f"禁止在管道函数内部直接调用另一个管道函数。\n"
        f"正确做法：将函数分离为 Pipeline 中的独立步骤。"
    )
    
    if n_ret == 0:
        @functools.wraps(func)
        def wrapper(ctx):
            assert isinstance(ctx, dict), _err_msg.format(actual_type=type(ctx).__name__)
            call_fn(ctx)
            return ctx
    elif n_ret == 1:
        _ret_key = ret_keys[0]
        @functools.wraps(func)
        def wrapper(ctx):
            assert isinstance(ctx, dict), _err_msg.format(actual_type=type(ctx).__name__)
            ctx[_ret_key] = call_fn(ctx)
            return ctx
    else:
        @functools.wraps(func)
        def wrapper(ctx):
            assert isinstance(ctx, dict), _err_msg.format(actual_type=type(ctx).__name__)
            result = call_fn(ctx)
            if not isinstance(result, tuple):
                result = (result,)
            assert len(ret_keys) == len(result), (
                f"{_fn_name} 声明返回 {len(ret_keys)} 个值 {ret_keys}，"
                f"实际返回 {len(result)} 个"
            )
            for k, v in zip(ret_keys, result):
                ctx[k] = v
            return ctx
    
    wrapper.fn = func
    wrapper.get = get_keys
    wrapper.ret = ret_keys
    wrapper._auto_ctx = True
    
    return wrapper


def auto_ctx(func):
    """参数名 → get，返回注解 → ret，非字符串注解不包装"""
    get_keys, ret_keys = _parse_annotation(func)
    
    if get_keys is None:
        return func
    
    return _build_ctx_wrapper(func, get_keys, ret_keys)
