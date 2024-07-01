"""
constants.py

This file is part of func2stream: Constants for logging and error messages.

Author: BI CHENG
GitHub: https://github.com/BICHENG/func2stream
License: MPL2.0
Created: 2024/7/1

For Usage, please refer to https://github.com/BICHENG/func2stream/samples or README.md
"""

# ------------------------ CHINESE ------------------------
# CORE.PY 
#   ASSERTION MESSAGES
ELEMENT_FN_NOT_CALLABLE = "[{}] 元素无法创建, 函数 {} 不是可调用的"
ELEMENT_KWARGS_NOT_DICT = "[{}] 元素无法创建, 参数 {} 不是字典类型"
ELEMENT_PARAMS_REQUIRED = "{}: 处理函数需要至少一个位置参数, 例如: def {}(item, ...)"
ELEMENT_FIRST_PARAM_NO_DEFAULT = "{}: 第一个位置参数 {} 不能有默认值"
ELEMENT_MISSING_PARAMS = "{}: 缺少{}个必需参数：{}，有效参数有：{}"
ELEMENT_EXTRA_KWARGS = "{}: 提供了{}个多余参数：{}"
ELEMENT_OTHER_NOT_INSTANCE = "{} 不是 Element 类的实例"
ELEMENT_THREAD_NOT_STARTED = "{} 元素没有启动, 无法处理元素"
ELEMENT_NO_SOURCE_QUEUE = "{} 元素没有设置输入队列, 无法启动"
ELEMENT_NO_SINK_QUEUE = "{} 元素没有设置输出队列, 无法启动"
ELEMENT_ALREADY_STARTED = "{} 元素已经启动, 请勿重复启动"
PIPELINE_TOO_FEW_ELEMENTS = "Pipeline 至少需要2个元素, 目前只有 {} 个"
PIPELINE_NO_SOURCE = "{}@{} 没有设置输入队列, 无法启动"
FUNC_CTX_NOT_DICT = "{}需要接收一个字典类型的上下文，但是接收到了 {} 类型，请检查上游的输出。"
FUNC_MISSING_KEYS = "{}需要获取的键: {} 不在上下文中，请检查上游的输出。"
FUNC_RETURN_MISMATCH = "{}返回的结果数量: {} 与设置的键({})数量: {} 不匹配，请检查函数的返回值。"
CTX_LOCALVARS_NOT_DICT = "需要在函数末尾提供一个上下文字典，但是没有找到，是否忘了 return locals()？"

#   LOGGING MESSAGES
SET_SRC_QUEUE = "设置 {} 的输入队列"
SET_DST_QUEUE = "设置 {} 的输出队列"
ELEMENT_STARTED = "已启动 {}"
ELEMENT_STOPPED = "已停止 {}"
ELEMENT_SINK_IS_LEAKY = "SINK {} 将是一个自动丢弃数据的队列"

ELEMENT_CONNECTION_BGN = "构建\t┌{} 的连接, 共 {} 个元素：┐"
ELEMENT_LINK_ESTABLISH = "\t│连接 {} -{}--> {} 已建立 [{}/{}]"
ELEMENT_CONNECTION_END = "\t└───────────────────"

EXEC_TIME_SUMMARY_HEADER = "{} 共 {} 个元素, 执行时间统计："
ELEMENT_EXEC_TIME_NAME = "\t{}执行时间："
ELEMENT_EXEC_TIME_AVG = "\t\t平均处理时间：{:.2f} ms"
ELEMENT_EXEC_TIME_MAX = "\t\t最大处理时间：{:.2f} ms"
ELEMENT_EXEC_TIME_MIN = "\t\t最小处理时间：{:.2f} ms"
ELEMENT_EXEC_TIME_TOP_5 = "\t\ttop 5% 处理时间：{:.2f} ms"
ELEMENT_EXEC_TIME_BTN_5 = "\t\tbottom 5% 处理时间：{:.2f} ms"
MOST_TIME_CONSUMING_HEADER = "\n\t最耗时的元素是 {}："

#   ERROR MESSAGES
DSOURCE_ERROR_OCCURRED = "{} 源发生异常：\n\t{} 位于 {}，参数：{}\n\ttraceback: {}"
ELEMENT_ERROR_OCCURRED = "{} 元素发生异常：\n\t{} 位于 {}，参数：{}\n\ttraceback: {}"


from loguru import logger

def assertl(condition, message):
    if not condition:
        logger.error(message)
        raise AssertionError(message)








